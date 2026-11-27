// The machine as a whole: construction, one core cycle (Machine::tick), run / done, the
// register drivers' order, and the host's read accessors.
//
// The machine's files (src/machine/), one hardware unit each:
//   machine.hpp        class Machine: every unit's state and this cycle's wires.
//   machine.cpp        the machine as a whole: construction, one core cycle (Machine::tick), run / done, the
//                      register drivers' order, and the host's read accessors.
//   launch_copy.cpp    the launch copy: the block the host placed in global memory is copied into the
//                      parameter area of shared memory while the warps already run.
//   front_end.cpp      the front end: fetch, the instruction memory's output and its decoder, packets,
//                      redirects.
//   branch.cpp         branch resolution (bra, bra.uni, ssy, sync) and the divergence stack.
//   issue.cpp          issue: admission, the scheduler, wait masks and depbar counters, accept, and the checks
//                      made when an instruction issues.
//   dispatch.cpp       dispatch: how an issued instruction reaches its unit. The dispatch pipeline and its end,
//                      where the tensor core and the accumulator engines take it (their dispatch_* functions are
//                      in their file), a dispatch entry's uniform operands; the memory queues of shared memory's
//                      two sides and the load / store unit (an entry's sample, its drop, the head leaving).
//   rf.cpp             the vector register file: the write calendar, the read and write port decisions over
//                      the async clients' requests (each client presents and takes its grant in its own
//                      file), the write ports' pipelines, the RAM stage and the predicate files. (Its timing
//                      rules are src/timing.hpp.)
//   urf.cpp            the uniform register file: its read rounds and its write pipeline.
//   arith.cpp          the arithmetic pipeline: its operand read, ALU, FPU and the special-function unit
//                      (black boxes with their contracts), their results at the RF write port.
//   shmem.cpp          shared memory: ports A and B, the two sides (queues, collect and serve stages), the
//                      write-back buffer, the selects, bank rounds, and what issue sees of them.
//   barriers.cpp       barriers (mbar): probes, the read-modify-write engine, atomics, and the try-wait slots.
//   tma.cpp            the TMA: its operand read, the command queue, the walker, requests and completions.
//   tma_ports.cpp      the TMA's rounds on the shared-memory ports: descriptor and store reads, placement
//                      writes, the sector register and the store registers.
//   tensor_core.cpp    the tensor core: the MMA engine on port B, tensor memory, and the accumulator transfers.
//   mma_numeric.cpp    the tensor core's arithmetic: what one step of the core computes (mma_numeric.hpp).
//   lsu.cpp            the load / store unit: its queue and command slots (their register reads and plans),
//                      requests, ingress, its register-file stages, the egress write port, and its register
//                      drivers.
//   gmem.cpp           the memory controller: the request channel, the queue, the bus, write acknowledgements.
//   sanitizer.cpp      the sanitizer (sanitizer.hpp): shadow state that only decides whether to PANIC.
//   sanitizer_arguments.cpp what the machine hands the sanitizer: pure reads of hardware state that feed
//                      nothing in the machine.
//   state_report.cpp   the state report: every piece of clocked state by unit, and the black boxes.
//   trace.cpp          the trace writer (trace.hpp).
//   common.cpp         helpers shared by the units: float bit conversions, round-robin grants, the lane mask and
//                      panics.
//   reg.hpp            Reg<T>, the clocked register every unit's state is made of.
//   regs.hpp           Regs<T>, a group of registers under one name: one per warp, per bank, per slot.
//   pipe.hpp           Pipe<T>, N stages in a row: what is in flight for a fixed number of cycles.
//   fifo.hpp           Fifo<T>, the deep queues toward global memory: push at the tail, pop at the head.
//   memory.hpp         Memory<T>, words in banks read and written only through their ports: the register
//                      files, shared memory, tmem, the instruction memory, the divergence stack.
//   blackbox.hpp       black boxes: units modelled opaquely behind a contract (registered inputs and outputs,
//                      a fixed latency); the memories are black boxes.
//   state.hpp          what a register may hold: a payload of a fixed size (checked when this compiles).
//   fixed.hpp          Bounded<T, N>, a payload field of at most N items: N places and a count.
//   internal.hpp       small pure functions that more than one unit uses (setp's comparison, a sector's
//                      bus beats, which unit takes an opcode). The other opcode classes are src/isa.hpp's.
//
// What the machine shares with the toolchain (src/, beside src/qtxc/) and which unit it goes with:
//   limits.hpp         the architectural maxima a profile may give the knobs that size state.
//   isa.hpp            the instruction set: opcodes, their classes, the instruction word. Every unit.
//   profile.hpp        the device profile: every size and delay. Every unit.
//   timing.hpp         every distance counted from an instruction's issue, and the read-port rule:
//                      issue.cpp, dispatch.cpp, rf.cpp, urf.cpp, arith.cpp, branch.cpp, and the operand
//                      reads of tma.cpp and tensor_core.cpp.
//   layout.hpp         where a tile's bytes sit in shared memory and tmem: tma.cpp, tma_ports.cpp,
//                      tensor_core.cpp.
//   tma_descriptor.hpp the TMA descriptor, its encoding and a tile's addresses: tma.cpp, tma_ports.cpp.
//   mbar.hpp           a barrier's eight bytes: barriers.cpp.

#include "machine/machine.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace nanosm {

Machine::Machine(Config config, Kernel kernel, Launch launch, TraceOptions trace_options)
    : config_(config),
      kernel_(std::move(kernel)) {
    try {
        validate_device_profile(config_);
    } catch (const ProfileError& error) {
        throw SimError(error.what());
    }
    // (validate_device_profile refused a profile with no lanes, banks, global
    // memory or sectors.)
    if (kernel_.register_count > 256 || config_.gmem_bytes > std::numeric_limits<std::size_t>::max() ||
        kernel_.words.empty()) {
        throw SimError("invalid zero-sized or unencodable machine resource");
    }
    const auto expected_profile = device_profile_fingerprint(config_);
    if (kernel_.profile_name != config_.name || kernel_.profile_fingerprint != expected_profile) {
        throw SimError(
            "kernel profile '" + kernel_.profile_name + "@" +
            format_profile_fingerprint(kernel_.profile_fingerprint) + "' does not match simulator '" +
            config_.name + "@" + format_profile_fingerprint(expected_profile) + "' (the fingerprint stamps the "
            "device profile and the instruction-set revision, " + std::to_string(isa_revision) + " here)");
    }
    // The rules every instruction must pass on this profile (src/qtxc/checks.cpp
    // check_instruction: what the profile supports, write masks, the
    // divergence stack, the one-round RF read (src/timing.hpp), mma.async
    // shapes, acc.ld / acc.st words per row, ldg.v8 / stg.v8 sectors) and
    // the uniform registers it names: refused here, at kernel load, in both
    // builds.
    check_kernel(kernel_, config_);
    // The host's part (src/image.hpp bind_launch): the warp count settled
    // against the kernel's `.warps`; the kernel's launch block (its `.data`,
    // its `.mbar` states, the launch's arguments) written into global memory
    // with the launch record saying where. The launch copy below brings it
    // into shared memory from the first cycle on.
    launch = bind_launch(kernel_, std::move(launch), config_);
    if (launch.active_warps == 0) throw SimError("active warp count is zero");
    // The SM's warp slots (core.warps): the scheduler's and the front end's
    // per-warp state exists for that many warps, whatever the RF could hold.
    if (launch.active_warps > config_.core_warps) {
        throw SimError("launch of " + std::to_string(launch.active_warps) + " warps exceeds the SM's " +
            std::to_string(config_.core_warps) + " warp slots (core.warps)");
    }
    for (uint32_t bank = 0; bank < config_.rf_banks; ++bank) {
        const auto rows_per_warp = kernel_.register_count <= bank
            ? 0
            : (kernel_.register_count - 1 - bank) / config_.rf_banks + 1;
        if (static_cast<uint64_t>(rows_per_warp) * launch.active_warps > config_.rf_rows_per_bank) {
            throw SimError(
                "kernel RF allocation cannot fit " + std::to_string(launch.active_warps) +
                " active warps (bank " + std::to_string(bank) + ")");
        }
    }
    // Bank 0 holds the most of a warp's rows (registers 0, banks, 2 x banks,
    // ...), so every warp takes that many rows in every bank (rf_address).
    rf_rows_per_warp_ = (kernel_.register_count + config_.rf_banks - 1) / config_.rf_banks;
    try {
        // The memories' shapes are beside them (machine.hpp).
        rf_.build(rf_shape());
        urf_.build(urf_shape());
        shmem_.build(shmem_shape());
        shmem_word_bytes_ = static_cast<uint32_t>(shmem_.shape().elements);
        gmem_.assign(static_cast<std::size_t>(config_.gmem_bytes), 0);
        tmem_.build(tmem_shape());
        acc_engines_.poke(AccEngines{});
        acc_ld_pipe_.build(config_.tmem_port_depth + config_.tmem_read_latency_cycles);
        acc_st_pipe_.build(config_.rf_read_latency_cycles + config_.tmem_port_depth);
        writeback_.build(config_.shmem_writeback_entries);
        shmem_counts_.build(config_.core_warps);
        a_read_pipe_.build(config_.shmem_port_a_depth + config_.shmem_read_latency_cycles);
        a_write_pipe_.build(config_.shmem_port_a_depth);
        b_write_pipe_.build(config_.shmem_port_a_depth);
        rmw_alu_.build(config_.shmem_rmw_compute_cycles);
        mma_reads_.build(config_.shmem_port_a_depth + config_.shmem_read_latency_cycles);
        mma_tags_.build(config_.mma_result_latency_cycles);
        mma_writes_.build(config_.tmem_port_depth);
        rf_write_policy_.build(config_.rf_banks);
        mbar_wait_slots_.build(config_.mbar_wait_slots);
        rf_read_policy_.build(config_.rf_banks);
        issue_read_banks_.build(config_.rf_banks);
        // The load buffer: 8 words per lane of a group. The store buffer the
        // same, and at least a word per lane (a scatter's data).
        const uint32_t load_words = 8 * config_.lsu_load_group_lanes;
        const uint32_t store_words = std::max(8 * config_.lsu_store_group_lanes, config_.lanes);
        lsu_buffers_.build(2);
        for (const auto& [index, words] : {std::pair<uint32_t, uint32_t>{load_buffer, load_words},
                                           std::pair<uint32_t, uint32_t>{store_buffer, store_words}}) {
            LsuBuffer empty;
            empty.words.assign(words, 0);
            empty.valid.assign(words, 0);
            lsu_buffers_.poke(index, std::move(empty));
        }
        free_id_tags_.build(config_.lsu_sectors_in_flight);
        lsu_read_pipe_.build(config_.rf_read_latency_cycles);
        lsu_collect_pipe_.build(config_.rf_read_latency_cycles);
        dispatch_pipe_.build(issue_to_dispatch(config_));
        arithmetic_reads_.build(issue_to_operands(config_));
        // The furthest booking: the slowest fixed result's write port.
        write_calendar_.build(issue_to_operands(config_) +
            std::max({config_.timing.alu, config_.timing.fadd, config_.timing.sfu}) + 1);
        // A booking d cycles ahead (1 .. stages - 1) enters at stage
        // stages - 1 - d and is in the stage after it from the edge on, so
        // stage 0 never holds one. Each other stage is a count of booked
        // ports per bank, zero meaning free (no valid bit).
        // The fixed units in the order of their delays (E each, src/timing.hpp;
        // their black boxes are built with their contracts, arith.cpp), and one
        // pipeline per RF write port, write_depth_cycles stages.
        const std::array<uint32_t, arithmetic_units> unit_delays{
            config_.timing.alu, config_.timing.fadd, config_.timing.sfu};
        for (std::size_t unit = 0; unit < arithmetic_units; ++unit) {
            units_slowest_first_[unit] = static_cast<ArithmeticUnit>(unit);
        }
        std::stable_sort(units_slowest_first_.begin(), units_slowest_first_.end(),
            [&](ArithmeticUnit a, ArithmeticUnit b) {
                return unit_delays[static_cast<std::size_t>(a)] > unit_delays[static_cast<std::size_t>(b)];
            });
        for (uint32_t bank = 0; bank < config_.rf_banks; ++bank) {
            for (uint32_t port = 0; port < config_.rf_write_ports_per_bank; ++port) {
                rf_write_pipes_.emplace_back(clocked_, Unit::register_file,
                    "RF write pipeline, bank " + std::to_string(bank) + " port " + std::to_string(port));
                rf_write_pipes_.back().build(config_.rf_write_depth_cycles);
            }
        }
        // One acknowledgement record per store that can be between its
        // dispatch and its completion. Such a store is in exactly one of:
        //   lsu.active_commands  its command slot, until its last sector
        //                        request is accepted;
        //   lsu.egress_sectors   requests all accepted, its last sector still
        //                        in the egress FIFO (there until its last
        //                        beat leaves the bus);
        //   the write window     its last beat left, its last acknowledgement
        //                        gmem.latency_dram_ticks later, the record
        //                        free two core cycles after that (the
        //                        completion's cycle, then free in Q): at most
        //                        one sector's last beat every beats-per-sector
        //                        ticks on the one bus.
        const auto ticks_per_cycle = static_cast<uint32_t>(
            (config_.dram_clock_hz + config_.core_clock_hz - 1) / config_.core_clock_hz);
        store_ack_records_detached_ = config_.lsu_egress_sectors + write_acks_in_flight_max(2 * ticks_per_cycle);
        store_acks_.build(config_.lsu_active_commands + store_ack_records_detached_);
        TmaSectorRegister sector;
        sector.data.assign(config_.gmem_sector_bytes, 0);
        tma_sector_.poke(std::move(sector));
        tma_store_registers_.build(config_.tma_store_sector_registers);
        // The FIFOs toward memory. The bus writes at most one beat per DRAM
        // tick, ceil(DRAM clock / core clock) ticks in a core cycle; the
        // ingress read port is a pluck wide, and the pluck may reach that
        // many bytes into the read records of its sectors (one more when it
        // starts inside one).
        const uint64_t dram_ticks_per_cycle = config_.dram_clock_hz / config_.core_clock_hz +
            (config_.dram_clock_hz % config_.core_clock_hz != 0 ? 1 : 0);
        const uint32_t pluck_bytes = 4 * config_.lsu_pluck_words_per_cycle;
        const std::size_t records_per_pluck = (pluck_bytes + config_.gmem_sector_bytes - 1) / config_.gmem_sector_bytes + 1;
        ingress_.build(std::size_t{config_.lsu_ingress_sectors} * config_.gmem_sector_bytes / 4,
            FifoPorts{static_cast<std::size_t>(dram_ticks_per_cycle * (config_.gmem_bus_bytes_per_dram_tick / 4)),
                config_.lsu_pluck_words_per_cycle, config_.lsu_ingress_latency_cycles});
        egress_open_.poke(std::nullopt);
        // The request pointer's FIFO shows its next entry too: the LSU's view
        // of the channel after a TMA write took the head (egress_request_slot).
        egress_requests_.build(config_.lsu_egress_sectors, FifoPorts{1, 2, config_.lsu_egress_latency_cycles});
        // The bus may finish a slot and start the next within one core cycle.
        egress_data_.build(config_.lsu_egress_sectors,
            FifoPorts{1, static_cast<std::size_t>(dram_ticks_per_cycle) + 1, config_.lsu_egress_latency_cycles});
        tma_reads_.build(config_.tma_sector_credits, FifoPorts{1, records_per_pluck, 1});
        lsu_reads_.build(std::size_t{config_.lsu_active_commands} * config_.lanes + 1, FifoPorts{1, records_per_pluck, 1});
        tma_returns_.build(config_.tma_sector_credits);
        tma_walker_start_pipe_.build(config_.tma_walker_start_cycles);
        warps_.reserve(launch.active_warps);
        for (uint32_t id = 0; id < launch.active_warps; ++id) {
            Warp warp;
            warp.id = id;
            warps_.push_back(std::move(warp));
        }
        predicates_.build(std::size_t{config_.core_warps} * predicates_per_warp);
        urf_pipe_.build(config_.urf_issue_to_port_cycles + config_.urf_write_depth_cycles);
        tma_read_pairs_.build(config_.urf_read_latency_cycles);
        // Every slot's FIFO empty and nothing in flight: launched warps fetch
        // from pc 0; a slot not launched stays halted.
        front_end_.build(config_.core_warps);
        for (uint32_t slot = launch.active_warps; slot < config_.core_warps; ++slot) {
            WarpFrontEnd idle;
            idle.halted = true;
            front_end_.poke(slot, idle);
        }
        // Every warp slot starts converged: all lanes active, an empty stack.
        divergence_.build(config_.core_warps, WarpDivergence{full_mask(), 0, {}});
        divergence_stack_.build(divergence_stack_shape());
        instruction_memory_.build(instruction_memory_shape());
        for (std::size_t word = 0; word < kernel_.words.size(); ++word) {
            instruction_memory_.poke(0, word / 2, word % 2, kernel_.words[word]);
        }
        // Per warp slot, launched or not (a slot not launched never counts).
        depbar_.build(config_.core_warps);
        credits_.build(credit_kinds);
        stall_left_.build(config_.core_warps);
        // The most one warp can have counted on one depbar. An instruction
        // counts from its admission (lds, lds.uni, sts, an atomic, ldg / stg
        // and their v8 forms, acc.ld / acc.st, mma.async; nothing else sets
        // a depbar) until its decrement, and all that time it is in exactly
        // one of these places, each of which holds only so many. A vector
        // memory form's read depbar changes nothing here: it names a counter
        // other than its set depbar, so one instruction counts at most once
        // on any counter, and it is lowered at the instruction's last vector
        // source read, which comes no later than its set depbar's decrement
        // (an lds / sts in its queue or collect stage, an ldg / stg / ldg.v8 /
        // stg.v8 in the LSU's queue or its command slot):
        //   shmem.read_queue_entries + shmem.write_queue_entries
        //                       an lds, lds.uni, sts or atomic in its queue
        //                       (one whose lanes are all off is decremented
        //                       there, at its issue + 1);
        //   4                   the two collect stages and the two serve
        //                       stages, one access each (an atomic until
        //                       its RMW starts);
        //   shmem.writeback_entries + 1
        //                       a load in its write-back entry, an atomic
        //                       in the old-word register, until the
        //                       register-file write decision grants its
        //                       write;
        //   shmem.port_a_depth  an sts's last round on its way to the RAM:
        //                       one round per A-write stage, decremented at
        //                       the last (it has left serve);
        //   lsu.queue_entries   an ldg or stg in the LSU's queue (one whose
        //                       lanes are all off is decremented there, at
        //                       its issue + 1);
        //   lsu.active_commands an ldg or stg in its command slot, a load's
        //                       until its last register write is granted, a
        //                       store's until its last sector request is
        //                       accepted;
        //   the store records beyond the command slots
        //                       a store out of its slot, until its last
        //                       acknowledgement's decrement (store_acks_,
        //                       above: lsu.egress_sectors + the write window);
        //   mma.queue_commands  an mma.async: decremented when its operands
        //                       are consumed, its command still in its slot;
        //   2                   an acc.ld and an acc.st: each engine, or the
        //                       claim on it (acc.st is decremented as it
        //                       frees its engine; acc.ld frees its engine
        //                       when its last read is launched);
        //   tmem.port_depth + tmem.read_latency_cycles
        //                       an acc.ld's last beat in the acc.ld pipeline
        //                       (one beat per stage, held there until its
        //                       register write is granted);
        //   rf.banks x rf.write_ports_per_bank x rf.write_depth_cycles
        //                       a load's last register write (lds, atomic,
        //                       ldg, acc.ld) between its grant and its
        //                       landing, where it is decremented: every
        //                       such write took a write port in its grant
        //                       cycle and is write_depth_cycles on the way;
        //   1                   a launch flag, counted from the launch until
        //                       the copy passes its cut.
        depbar_counter_max_ = config_.shmem_read_queue_entries + config_.shmem_write_queue_entries + 4 +
            config_.shmem_writeback_entries + 1 + config_.shmem_port_a_depth +
            config_.lsu_queue_entries + config_.lsu_active_commands + store_ack_records_detached_ +
            config_.mma_queue_commands + 2 +
            config_.tmem_port_depth + config_.tmem_read_latency_cycles +
            config_.rf_banks * config_.rf_write_ports_per_bank * config_.rf_write_depth_cycles + 1;
        declare_black_boxes();
        sanitizer_.record_launch(config_, kernel_, launch.active_warps);
    } catch (const std::bad_alloc&) {
        throw SimError("machine state does not fit in host memory");
    } catch (const std::length_error&) {
        throw SimError("machine state exceeds host container limits");
    }
    trace_.emplace(
        launch.active_warps, config_.core_clock_hz, config_.dram_clock_hz, trace_options);
    trace_->set_mma_shapes(config_.mma_shapes);
    for (const auto& segment : launch.test_shmem) {
        if (segment.address > shmem_bytes() || segment.bytes.size() > shmem_bytes() - segment.address) {
            throw SimError("launch shmem image segment does not fit");
        }
        for (std::size_t byte = 0; byte < segment.bytes.size(); ++byte) {
            const auto at = shmem_word(static_cast<uint32_t>(segment.address + byte));
            shmem_.poke(at.bank, at.row, at.byte, segment.bytes[byte]);
        }
    }
    uint64_t test_preload_bytes = 0;
    for (const auto& segment : launch.test_shmem) test_preload_bytes += segment.bytes.size();
    trace_->test_preload(test_preload_bytes);
    apply_image(gmem_, launch.gmem_image, "gmem");
    // The mbars the launch declares (the kernel's `.mbar` sections, and a
    // test's own, initialized by hand in the shmem
    // preload): live from launch for the sanitizer's live-mbar check,
    // like a SIMT mbar.init's. A declaration is metadata;
    // the machine only checks that it names an mbar slot.
    for (const auto address : launch.mbars) {
        if (address % 8 != 0 || static_cast<uint64_t>(address) + 8 > shmem_bytes()) {
            throw SimError("launch declares an mbar at " + std::to_string(address) +
                ", which is not 8-byte aligned in shmem");
        }
    }
    sanitizer_.record_launch_mbars(launch.mbars);
    if (launch.record.has_value()) start_launch_copy(*launch.record);
}

void Machine::set_driver_shuffle_seed(uint64_t seed) {
    shuffle_.emplace(seed);
}

// ===========================================================================
// One core cycle.
//
// 1. WIRES. Every decision of the cycle is computed once, in the order below.
//    A decision reads register Q values (cur()), memories, and the wires
//    above it: requests flow toward the shared resource, the resource's one
//    select function grants, data flows downstream. No register's next value
//    is visible here. Same-cycle visibility exists only where a wire carries
//    it. The declared ones:
//      (a) None through the RF, the URF or the predicate file (src/timing.hpp):
//          a write lands at the clock edge of its RAM cycle, after every
//          read of that cycle (READ_FIRST); no same-edge forwarding.
//      (b) None from a completion to issue: a depbar clear, an mbar wait
//          slot's completion (wake, immediate success, timeout), a probe's
//          return reach issue through their registers, the next cycle.
//      (c) None into issue's admission: every question it asks reads Q
//          (admission_refusal). A queue entry, an engine, a slot or an order
//          freed at t is seen from t + 1 -- the credits of the queues and
//          engines (a credit is held from the accept, through the dispatch
//          pipeline or the TMA operand read, until the unit frees the
//          entry), a warp's shared-memory counts and probe bit, fence.sh's
//          count.
//          A bank's read and write inhibits are bits of the RF policy
//          registers, set from the RF decisions of the cycle before: an
//          async writer denied with service owed at t, or a reader denied
//          at t while the instruction accepted at t takes every read port
//          of its bank, refuses issue's use of the bank from t + 1.
//      (d) Same-cycle space reuse: a register drained at t may be refilled at
//          t (the load and store buffers, a shared-memory collect or serve
//          stage whose access moves on at t, the TMA sector / store
//          registers, the MMA event register, an LSU command slot taken by
//          the LSU queue's head as the RF write grant releases it). The
//          refill's data is visible at t+1; nothing passes through. (Not the
//          write-back buffer or the old-word register: an entry whose RF
//          write is granted at t is free from t + 1. Not an acc engine: its
//          next transfer issues after it is empty in Q.)
//      (e) (None: a stg.v8 reads its data only once its command is planned,
//          from Q.)
//      (f) Inside the MMA: a step whose tile's previous result lands at this
//          cycle's edge may start its operand fetch now; its C is read from
//          tmem cycles after that landing (tmem itself is read first).
//      (g) The placeholder controller may put a request accepted at t on the
//          bus at t (outside the SM; the timeline test pins request -> beat 0).
//      (h) A TMA shmem transaction is granted its first round in the cycle
//          it is requested: a declared combinational path from latched TMA
//          state (its queue and sector registers, Q; for a store sector also
//          this cycle's egress drain of a store register (d)) ->
//          tma_requests builds the request -> select_a_read / select_b_write
//          grants it -> round 0 enters its pipeline, all in one cycle. No
//          register holds the transaction before round 0. Likewise an RMW's
//          start (A-write's alternation lets it begin, the engine takes the
//          request from its register) and its read grant may share a cycle:
//          select_a_read sees the start (shmem wire order: A-write, RMW
//          admission, A-read). A SIMT access or RMW enters its side's serve
//          stage at an edge and is served from the next cycle. A new MMA
//          fetch serves its first port-B round in the cycle it starts.
//      (i) None from issue into fetch: the fetch decision reads registers
//          only (the
//          FIFOs' valid words, the read in flight, halted, the issue
//          register's branch). A slot an issue frees at t is seen by fetch
//          at t + 1; a branch accepted at t blocks its warp's fetch from
//          t + 1. Issue sees the FIFO heads' Q: a word landing at t issues
//          from t + 1.
//      (j) A zero fill of the TMA walker goes only in a cycle the ingress
//          completes no TMA sector (the placement tickets' one write port):
//          the ingress read decision -> the walker's step, in one cycle.
// 2. DRIVERS. Each register's next value, from Q and the wires. They run in
//    any order (set_driver_shuffle_seed() permutes them every cycle).
// 3. CLOCK EDGE. The writes at the memories' RAM this cycle go through their
//    write ports (src/machine/memory.hpp); then every register takes D and
//    every memory its writes (shmem: A-write's, then port-B write's).
// ===========================================================================
void Machine::tick() {
    if (done()) return;
    sanitizer_.record_cycle_start();
    Wires w;

    // The front end, from Q only (i): the fetch grant, the word at the
    // instruction memory's output, the control instruction in the issue
    // register resolved (with its divergence-stack push or pop and the warp's
    // next active mask), and the stack memory's output (a pop's refill).
    w.fetch = decide_fetch();
    w.fetched = instruction_memory_output();
    w.branch = resolve_branch();
    w.divergence_refilled = divergence_memory_output();

    // The RF write path, from Q: each async writer presents its write (the
    // write-back entries and the old-word register of shared memory, the load
    // / store unit's load buffer, the acc.ld pipeline's output); the one RF
    // write decision shares this cycle's unbooked write ports among them;
    // the write port (the arithmetic units' booked results and the granted
    // writes enter their ports' write pipelines); the load / store unit takes
    // its grant; the RAM stage (the writes landing at this cycle's edge, and
    // their completions). Then the other completions presented from Q now: a
    // store whose last acknowledgement came last cycle, a setp's predicate at
    // the ALU's or the FPU's output.
    RfWriteRequests rf_writes;
    rf_writes.writeback = writeback_rf_writes();
    rf_writes.atomic_return = atomic_return_rf_write();
    rf_writes.load_buffer = load_buffer_rf_write();
    rf_writes.acc_ld = acc_ld_rf_write();
    w.rf_write = decide_rf_writes(rf_writes);
    rf_write_port(arithmetic_rf_writes(), rf_writes, w);
    take_load_buffer_grant(rf_writes.load_buffer, w);
    rf_ram_stage(w);
    complete_stores(w);
    present_setp_predicate(w);
    // The URF write pipeline's write port and RAM stage, from Q: nothing
    // contends for them (one writer per stage by construction).
    urf_ram_stage(w);

    // The store path's last stage frees a TMA store register / the store
    // buffer for a refill this same cycle (d).
    w.egress = decide_egress_write();

    // Request generation: independent of each other.
    if (shuffle_.has_value() && ((*shuffle_)() & 1u) != 0) {
        w.lsu_want = lsu_request_want();
        w.tma = tma_requests(w.egress);
    } else {
        w.tma = tma_requests(w.egress);
        w.lsu_want = lsu_request_want();
    }

    // The shared resources, one select each: the request channel; shmem's
    // A-read, A-write and port-B write (after the RAM stage: the write rounds
    // reaching the RAM and their completions, the read round at the port
    // output and its requester).
    w.channel = select_request_channel(w.tma, w.lsu_want);
    step_shmem(w);

    // The MMA and the acc engines, each on a copy of its Q. The MMA's commit
    // may present into the event register the RMW engine claimed just now (d).
    step_mma(w);
    step_acc_transfers(w);        // acc.ld advances on its RF grant
    decide_mbar_timeouts(w);      // a wake this cycle wins over the deadline
    w.ingress = decide_ingress(w);
    step_gmem_controller(w);
    step_tma(w);

    // The TMA operand read's end: a tma's last words at the URF output now
    // push its command into the TMA queue. Then the dispatch pipeline's end:
    // an engine instruction whose operands arrive now goes to its unit. The
    // LSU's queue: its newest entry's issue + 1 sample, its head into a
    // command slot (one the RF write grant released now included (d)), the
    // plan of the address at the RF output now.
    arrive_tma_operands(w);
    dispatch_issued(w);
    step_lsu_queue(w);
    // The async RF readers present their reads (the acc.st engine's beat, the
    // LSU's stg.v8 data beat, the shared-memory collect stages, the LSU's
    // command slots: from Q and the egress decision); the RF read decision
    // grants them the read ports the issue register's round leaves free now
    // and reads them; each reader takes its grant, in that order.
    RfReadRequests rf_reads;
    rf_reads[RfReader::acc_st] = acc_st_rf_read();
    rf_reads[RfReader::lsu_store] = lsu_store_rf_read(w.egress);
    rf_reads[RfReader::read_collect] = collect_rf_read(MemoryQueue::shmem_read);
    rf_reads[RfReader::write_collect] = collect_rf_read(MemoryQueue::shmem_write);
    rf_reads[RfReader::lsu_collect] = lsu_collect_rf_read();
    w.rf_read = decide_rf_reads(rf_reads, issue_read_banks_.cur());
    take_acc_st_read(w);
    take_lsu_store_read(w);
    take_collect_reads(w);
    take_lsu_collect_read(w);

    // The scheduler. Its admission reads Q only (c): no decision above
    // reaches it. Its write booking is in a future cycle and its read round
    // is the next cycle's (the issue register); with the RF decisions above
    // it sets the owed and inhibit bits' next state.
    decide_issue(w);

    // The front ends' next state: issue's pop (and halt), the landing word,
    // the redirect, the fetch grant -- one calculation per warp.
    step_front_end(w);

    // The RF read stage: the issue register's round samples now (vector and
    // uniform registers), each unit's stage 0 its own operands -- the
    // arithmetic pipeline (whose last stage executes now, its operands at the
    // RF output), the shared-memory queues' newest entry, the dispatch
    // pipeline, the URF write pipeline -- and the round counter its round;
    // then the read ports used are checked. Then the URF write pipeline's work
    // by stage: its sampled write, the uniform ALU and r2ur's results, issue's
    // new write.
    step_arithmetic_reads(w);
    sample_newest(MemoryQueue::shmem_read, w);
    sample_newest(MemoryQueue::shmem_write, w);
    sample_dispatch_round(w);
    sample_urf_pipe(w);
    sample_urf_rounds(w);
    check_urf_round_reads(w);
    check_round_reads(w);
    step_urf(w);

    run_drivers(w);
    // The memories' writes at their RAM now, through their write ports (the
    // RF's, the URF's, shared memory's ports A and B, tmem's, the divergence
    // stack's). Each memory holds them until the clock edge.
    land_rf_writes();
    land_urf_write(w);
    land_shmem_writes(w);
    land_tmem_writes(w);
    land_divergence_push(w);

    // Clock edge: every register takes D and every memory its writes.
    clocked_.commit();
    ++cycle_;
    // Kernel end: the sanitizer's end-of-kernel checks (a nonzero mbar tx
    // count). Every unit is idle then, so every credit has been given back
    // and every depbar lowered as often as it was raised (a set depbar, a
    // read depbar, a launch flag).
    if (done()) {
        sanitizer_.check_kernel_end(cycle_);
        for (std::size_t k = 0; k < credit_kinds; ++k) {
            if (const auto held = credits_.cur()[k]; held != 0) {
                throw SimError("internal: the kernel ends with " + std::to_string(held) + " of " +
                    credit_name(static_cast<Credit>(k)) + "'s credits held (taken more often than given back)");
            }
        }
        for (uint32_t warp = 0; warp < warps_.size(); ++warp) {
            for (uint8_t bar = 0; bar < 6; ++bar) {
                if (const auto value = depbar_.cur().at(warp).at(bar); value != 0) {
                    throw SimError("internal: warp " + std::to_string(warp) + " ends the kernel with depbar sb" +
                        std::to_string(bar) + " at " + std::to_string(value) + " (raised more often than lowered)");
                }
            }
        }
    }
}

void Machine::run(uint64_t cycle_limit) {
    while (!done() && cycle_ < cycle_limit) tick();
    if (!done()) throw SimError("cycle cap at cycle " + std::to_string(cycle_));
    trace_->finish(cycle_);
}

bool Machine::done() const {
    // Every warp halted and no instruction-memory read in flight (its word
    // would be dropped, but the read is still under way).
    const auto& front_end = front_end_.cur();
    const bool warps_halted = std::all_of(front_end.begin(), front_end.end(), [](const WarpFrontEnd& warp) {
        return warp.halted;
    }) && !fetch_read_.cur().has_value() && !divergence_refill_.cur().has_value();
    const auto& waits = mbar_wait_slots_.cur();
    const bool wait_slots_empty = std::all_of(
        waits.begin(), waits.end(), [](const auto& slot) { return !slot.has_value(); });
    const auto& tma = tma_.cur();
    const auto& mma = mma_.cur();
    const auto& acc = acc_engines_.cur();
    const auto empty = [](const auto& reg) { return !reg.cur().has_value(); };
    const auto all_empty = [](const auto& group) {
        return std::all_of(group.cur().begin(), group.cur().end(), [](const auto& slot) { return !slot.has_value(); });
    };
    // The three memory queues. Shared memory's two sides (collect and serve
    // stages), the write-back buffer and the old-word register, the RMW
    // engine (its operation and ALU stages) and the services' pipelines (a
    // store that has left serve may still be travelling to its write edge).
    const bool queues_empty = std::all_of(std::begin(queues_), std::end(queues_), [](const auto& queue) {
        return queue.cur().empty();
    });
    const bool shmem_idle = std::all_of(std::begin(collects_), std::end(collects_), empty) && empty(read_serve_) &&
        empty(write_serve_) && all_empty(writeback_) && empty(atomic_return_) &&
        !rmw_.cur().active.has_value() && rmw_alu_.empty() && a_read_pipe_.empty() &&
        a_write_pipe_.empty() && b_write_pipe_.empty() && empty(tma_read_txn_) &&
        empty(tma_write_txn_);
    return warps_halted && wait_slots_empty && queues_empty && shmem_idle && acc_ld_pipe_.empty() && acc_st_pipe_.empty() &&
        !tma_mbar_event_.cur().has_value() && !mma_mbar_event_.cur().has_value() && gmem_queue_.empty() &&
        !gmem_bus_.has_value() && pending_write_acks_.empty() &&
        lsu_commands_.cur().empty() && lsu_collect_pipe_.empty() &&
        tma_reads_.empty() && lsu_reads_.empty() && ingress_.empty() && empty(egress_open_) &&
        egress_requests_.empty() && egress_data_.empty() &&
        !tma_sector_.cur().request.has_value() && all_empty(tma_store_registers_) &&
        tma.commands.empty() && tma_returns_.empty() && !tma.event_command.has_value() &&
        tma.credits_used == 0 && tma_walker_start_pipe_.empty() && empty(urf_read_rounds_) &&
        tma_read_pairs_.empty() &&
        empty(tma_holding_) && arithmetic_reads_.empty() &&
        std::all_of(arithmetic_black_boxes_.begin(), arithmetic_black_boxes_.end(),
            [](const auto& black_box) { return black_box.empty(); }) &&
        std::all_of(rf_write_pipes_.begin(), rf_write_pipes_.end(), [](const auto& pipe) { return pipe.empty(); }) &&
        std::all_of(store_acks_.cur().begin(), store_acks_.cur().end(),
            [](const StoreAckRecord& record) { return !record.valid; }) &&
        dispatch_pipe_.empty() && lsu_read_pipe_.empty() && urf_pipe_.empty() &&
        mma.slots.empty() && mma_core_.empty() && mma_tags_.empty() && mma_writes_.empty() && !mma.fetch.has_value() &&
        mma_reads_.empty() &&
        !acc[0].has_value() && !acc[1].has_value();
}

uint32_t Machine::read_shmem_u32(uint32_t byte_address) const {
    if (byte_address % 4 != 0 || static_cast<uint64_t>(byte_address) + 4 > shmem_bytes()) {
        throw SimError("invalid host shmem u32 read");
    }
    uint32_t value = 0;
    std::memcpy(&value, shmem_.contents().data() + byte_address, sizeof(value));
    return value;
}

uint32_t Machine::read_gmem_u32(uint64_t byte_address) const {
    if (byte_address % 4 != 0 || byte_address + 4 < byte_address || byte_address + 4 > gmem_.size()) {
        throw SimError("invalid host gmem u32 read");
    }
    uint32_t value = 0;
    std::memcpy(&value, gmem_.data() + static_cast<std::size_t>(byte_address), sizeof(value));
    return value;
}

MbarState Machine::read_mbar_state(uint32_t byte_address) const {
    return load_mbar(byte_address);
}

void Machine::complete_mbar_tx(uint32_t byte_address, uint32_t bytes, uint8_t parity) {
    if (byte_address % 8 != 0 || static_cast<uint64_t>(byte_address) + 8 > shmem_bytes()) {
        throw SimError("invalid engine mbar address");
    }
    if (parity > 1) throw SimError("invalid engine mbar parity");
    // A DMA complete_tx: it belongs in the TMA's event register. It changes the
    // mbar's current phase; `parity` (the phase the host means) is sanitizer
    // metadata only.
    if (tma_mbar_event_.cur().has_value()) {
        throw SimError("TMA mbar event register is occupied; an engine holds one event at a time");
    }
    MbarRmwRequest request;
    request.operation = MbarRmwOperation::complete_tx;
    request.byte_address = byte_address;
    request.value = bytes;
    request.parity = parity;
    tma_mbar_event_.poke(std::move(request)); // between ticks: held from the next cycle on
}

uint32_t Machine::read_reg(
    uint32_t warp, uint32_t lane, uint32_t reg) const {
    if (warp >= warps_.size() || lane >= config_.lanes || reg >= kernel_.register_count) {
        throw SimError("invalid host RF read");
    }
    const auto address = rf_address(warp, reg);
    return rf_.peek(address.bank, address.row)[lane];
}

uint32_t Machine::read_ureg(uint32_t warp, uint32_t reg) const {
    if (warp >= warps_.size() || reg >= config_.urf_registers) throw SimError("invalid host URF read");
    return urf_.peek(0, urf_row(warp, reg))[0];
}

// ===========================================================================
// Register drivers: the order they run in (each drive_* function is in its
// unit's file). Each computes its register's next value from Q and the
// wires, and writes it once (or not at all: hold). They read no other
// register's D and no state another driver writes, so they run in any order;
// set_driver_shuffle_seed() permutes them every cycle as a regression check.
// Simultaneous events combine in one calculation (a release and a
// reallocation, a pop and a push, completions and an admission); the
// one-write rule is never a reason to defer an event. Inside a calculation
// the old owner's drains, clears and release come before the new owner's
// install and fills, so a same-cycle reuse (d) cannot erase the new state.
// ===========================================================================

void Machine::run_drivers(const Wires& w) {
    static constexpr std::array<Driver, 49> canonical = {
        &Machine::drive_front_end,
        &Machine::drive_fetch,
        &Machine::drive_issue_branch,
        &Machine::drive_divergence,
        &Machine::drive_depbar,
        &Machine::drive_scheduler,
        &Machine::drive_credits,
        &Machine::drive_wait_slots,
        &Machine::drive_queues,
        &Machine::drive_shmem_collects,
        &Machine::drive_shmem_serves,
        &Machine::drive_writeback,
        &Machine::drive_atomic_return,
        &Machine::drive_shmem_counts,
        &Machine::drive_rmw_engine,
        &Machine::drive_a_read_pipe,
        &Machine::drive_write_pipes,
        &Machine::drive_shmem_arbiters,
        &Machine::drive_tma_event,
        &Machine::drive_mma_event,
        &Machine::drive_mma,
        &Machine::drive_acc_transfers,
        &Machine::drive_acc_ld_pipe,
        &Machine::drive_tma_queue,
        &Machine::drive_tma_walker_start,
        &Machine::drive_tma_transactions,
        &Machine::drive_tma_sector,
        &Machine::drive_tma_store_registers,
        &Machine::drive_tma_returns,
        &Machine::drive_tma_operand_read,
        &Machine::drive_lsu_collect_pipe,
        &Machine::drive_lsu_commands,
        &Machine::drive_lsu_buffers,
        &Machine::drive_ingress,
        &Machine::drive_egress,
        &Machine::drive_read_records,
        &Machine::drive_free_id_tags,
        &Machine::drive_request_channel,
        &Machine::drive_rf_write_pipes,
        &Machine::drive_store_acks,
        &Machine::drive_predicates,
        &Machine::drive_rf_write_policy,
        &Machine::drive_rf_read_policy,
        &Machine::drive_issue_read_banks,
        &Machine::drive_dispatch_pipe,
        &Machine::drive_arithmetic_reads,
        &Machine::drive_write_calendar,
        &Machine::drive_lsu_read_pipe,
        &Machine::drive_urf_pipe,
    };
    auto order = canonical;
    if (shuffle_.has_value()) std::shuffle(order.begin(), order.end(), *shuffle_);
    for (const auto driver : order) (this->*driver)(w);
}

void Machine::apply_image(
    std::vector<uint8_t>& memory,
    const std::vector<ImageSegment>& image,
    const std::string& name) {
    for (const auto& segment : image) {
        if (segment.address > memory.size() || segment.bytes.size() > memory.size() - segment.address) {
            throw SimError("launch " + name + " image segment does not fit");
        }
        std::copy(
            segment.bytes.begin(),
            segment.bytes.end(),
            memory.begin() + static_cast<std::size_t>(segment.address));
    }
}

} // namespace nanosm
