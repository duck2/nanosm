// The machine as a whole: construction, one core cycle (Machine::tick), run / done, the
// register drivers' order, and the host's read accessors.
//
// The machine's files (src/machine/), one hardware unit each:
//   machine.hpp        class Machine: every unit's state and this cycle's wires.
//   machine.cpp        the machine as a whole: construction, one core cycle (Machine::tick), run / done, the
//                      register drivers' order, and the host's read accessors.
//   launch_copy.cpp    the launch copy: the block the host placed in global memory is copied into the
//                      parameter area of shared memory while the warps already run.
//   front_end.cpp      the front end: fetch, the instruction memory's output, packets, redirects.
//   branch.cpp         branch resolution (bra, bra.uni, ssy, sync) and the divergence stack.
//   issue.cpp          issue: admission, the scheduler, wait masks and depbar counters, accept, and the checks
//                      made when an instruction issues.
//   dispatch.cpp       dispatch: the dispatch pipeline and its end, where each unit takes an issued instruction
//                      (a unit's own dispatch_* function is in the unit's file), and a dispatch entry's operands.
//   rf.cpp             the vector register file: the write calendar, the read and write port decisions, the
//                      RAM stages, and writebacks. (Its timing rules are src/timing.hpp.)
//   urf.cpp            the uniform register file: its read rounds and its write pipeline.
//   arith.cpp          the arithmetic pipeline: ALU, FPU and the special-function unit.
//   shmem.cpp          shared memory: ports A and B, the selects, load and store slots, bank rounds, and what
//                      issue sees of them.
//   barriers.cpp       barriers (mbar): probes, the read-modify-write engine, atomics, and the try-wait slots.
//   tma.cpp            the TMA: its operand read, the command queue, the walker, requests and completions.
//   tma_ports.cpp      the TMA's rounds on the shared-memory ports: descriptor and store reads, placement
//                      writes, the sector register and the store registers.
//   tensor_core.cpp    the tensor core: the MMA engine on port B, tensor memory, and the accumulator transfers.
//   mma_numeric.cpp    the tensor core's arithmetic: what one step of the core computes (mma_numeric.hpp).
//   lsu.cpp            the load / store unit: admission and requests, ingress, its register-file stages, the
//                      egress write port, how a command enters (dispatch_lsu), and its register drivers.
//   gmem.cpp           the memory controller: the request channel, the queue, the bus, write acknowledgements.
//   sanitizer.cpp      the sanitizer (sanitizer.hpp): shadow state that only decides whether to PANIC.
//   sanitizer_arguments.cpp what the machine hands the sanitizer: pure reads of hardware state that feed
//                      nothing in the machine.
//   trace.cpp          the trace writer (trace.hpp).
//   common.cpp         helpers shared by the units: float bit conversions, round-robin grants, the lane mask and
//                      panics.
//   reg.hpp            Reg<T>, the clocked register every unit's state is made of.
//   pipe.hpp           Pipe<T>, N stages in a row: what is in flight for a fixed number of cycles.
//   internal.hpp       small pure functions that more than one unit uses (setp's comparison, a sector's
//                      bus beats). The opcode classes are src/isa.hpp's.
//
// What the machine shares with the toolchain (src/, beside src/qtxc/) and which unit it goes with:
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
    if (config_.lanes == 0 || config_.rf_banks == 0 || config_.shmem_banks == 0 ||
        config_.gmem_bytes == 0 || kernel_.register_count > 256 ||
        config_.gmem_bytes > std::numeric_limits<std::size_t>::max() ||
        config_.gmem_sector_bytes == 0 || config_.gmem_bus_bytes_per_dram_tick == 0 ||
        config_.gmem_sector_bytes % 4 != 0 || config_.gmem_queue_sectors == 0 ||
        kernel_.words.empty()) {
        throw SimError("invalid zero-sized or unencodable machine resource");
    }
    const auto expected_profile = device_profile_fingerprint(config_);
    if (kernel_.profile_name != config_.name || kernel_.profile_fingerprint != expected_profile) {
        throw SimError(
            "kernel profile '" + kernel_.profile_name + "@" +
            format_profile_fingerprint(kernel_.profile_fingerprint) + "' does not match simulator '" +
            config_.name + "@" + format_profile_fingerprint(expected_profile) + "'");
    }
    if (config_.mbar_wait_slots == 0) {
        for (const auto word : kernel_.words) {
            if (is_mbar_try_wait(decode(word).opcode)) {
                throw SimError("mbar.try_wait is unsupported by this device profile");
            }
        }
    }
    if (config_.tma_queue_commands == 0) {
        for (const auto word : kernel_.words) {
            if (is_tma_opcode(decode(word).opcode)) {
                throw SimError("tma.load / tma.store are unsupported by this device profile");
            }
        }
    }
    // ssy, sync and a predicated bra need the divergence stack
    // (core.divergence_stack_entries).
    check_divergence_stacks(kernel_, config_);
    // Distinct sources beyond a bank's read ports are a property of the
    // instruction (src/timing.hpp): refused here, at kernel load, in both builds.
    check_rf_read_rounds(kernel_, config_);
    // A write mask only on an instruction that can be masked (src/isa.hpp
    // can_be_masked).
    check_write_masks(kernel_, config_);
    // An mma.async's shape is one the profile lists; an acc.ld / acc.st's
    // words per row are legal (src/layout.hpp).
    check_mma_shapes(kernel_, config_);
    // An ldg.v8 / stg.v8 needs 32-byte sectors.
    check_v8_sectors(kernel_, config_);
    // Every uniform register named is one the profile's URF has.
    check_uniform_registers(kernel_, config_);
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
    try {
        shmem_.assign(config_.shmem_bytes, 0);
        gmem_.assign(static_cast<std::size_t>(config_.gmem_bytes), 0);
        tmem_.assign(static_cast<std::size_t>(config_.tmem_fine_banks) * config_.tmem_words_per_fine_bank, 0);
        acc_engines_.poke(AccEngines{});
        acc_ld_pipe_.build(config_.tmem_port_depth + config_.tmem_read_latency_cycles);
        acc_st_pipe_.build(config_.rf_read_latency_cycles + config_.tmem_port_depth);
        for (uint32_t index = 0; index < config_.shmem_load_slots; ++index) {
            load_slots_.emplace_back(registers_, "load slot " + std::to_string(index));
        }
        for (uint32_t index = 0; index < config_.shmem_store_slots; ++index) {
            store_slots_.emplace_back(registers_, "store slot " + std::to_string(index));
        }
        a_read_pipe_.build(config_.shmem_port_a_depth + config_.shmem_read_latency_cycles);
        a_write_pipe_.build(config_.shmem_port_a_depth);
        b_write_pipe_.build(config_.shmem_port_a_depth);
        rmw_alu_.build(config_.shmem_rmw_compute_cycles);
        mma_reads_.build(config_.shmem_port_a_depth + config_.shmem_read_latency_cycles);
        mma_results_.build(config_.mma_result_latency_cycles + config_.tmem_port_depth);
        RfWritePolicy policy;
        policy.owed.assign(config_.rf_banks, 0);
        policy.cursor.assign(config_.rf_banks, 0);
        rf_write_policy_.poke(std::move(policy));
        mbar_wait_slots_.poke(WaitSlots(config_.mbar_wait_slots));
        RfReadPolicy read_policy;
        read_policy.owed.assign(config_.rf_banks, 0);
        read_policy.cursor.assign(config_.rf_banks, 0);
        rf_read_policy_.poke(std::move(read_policy));
        issue_read_banks_.poke(std::vector<uint8_t>(config_.rf_banks, 0));
        // The load buffer: 8 words per lane of a group. The store buffer the
        // same, and at least a word per lane (a scatter's data).
        const uint32_t load_words = 8 * config_.lsu_load_group_lanes;
        const uint32_t store_words = std::max(8 * config_.lsu_store_group_lanes, config_.lanes);
        for (const auto& [name, words] : {std::pair<const char*, uint32_t>{"LSU load buffer", load_words},
                                          std::pair<const char*, uint32_t>{"LSU store buffer", store_words}}) {
            LsuBuffer empty;
            empty.words.assign(words, 0);
            empty.valid.assign(words, 0);
            lsu_buffers_.emplace_back(registers_, name, std::move(empty));
        }
        free_id_tags_.poke(std::vector<uint8_t>(config_.lsu_sectors_in_flight, 0));
        lsu_read_pipe_.build(config_.rf_read_latency_cycles);
        dispatch_pipe_.build(issue_to_dispatch(config_));
        arithmetic_reads_.build(issue_to_operands(config_));
        // The furthest booking: the slowest fixed result's write port.
        write_calendar_.build(issue_to_operands(config_) +
            std::max({config_.timing.alu, config_.timing.fadd, config_.timing.sfu}) + 1);
        write_calendar_.declare_payload_bits(
            uint64_t{config_.rf_banks} * static_cast<uint64_t>(std::bit_width(config_.rf_write_ports_per_bank)));
        TmaSectorRegister sector;
        sector.data.assign(config_.gmem_sector_bytes, 0);
        tma_sector_.poke(std::move(sector));
        TmaStoreRegisters stores;
        stores.ring.resize(config_.tma_store_sector_registers);
        tma_store_registers_.poke(std::move(stores));
        tma_walker_start_pipe_.build(config_.tma_walker_start_cycles);
        warps_.reserve(launch.active_warps);
        for (uint32_t id = 0; id < launch.active_warps; ++id) {
            Warp warp;
            warp.id = id;
            warp.rf.resize(static_cast<std::size_t>(config_.lanes) * kernel_.register_count, 0);
            for (auto& predicate : warp.predicates) predicate.resize(config_.lanes, 0);
            warp.urf.resize(config_.urf_registers, 0);
            warps_.push_back(std::move(warp));
        }
        urf_pipe_.build(config_.urf_issue_to_port_cycles + config_.urf_write_depth_cycles);
        tma_read_pairs_.build(config_.urf_read_latency_cycles);
        // Every slot's FIFO empty and nothing in flight: launched warps fetch
        // from pc 0; a slot not launched stays halted.
        std::vector<WarpFrontEnd> front_end(config_.core_warps);
        for (uint32_t slot = launch.active_warps; slot < config_.core_warps; ++slot) front_end[slot].halted = true;
        front_end_.poke(std::move(front_end));
        // Every warp slot starts converged: all lanes active, an empty stack.
        DivergenceState divergence;
        divergence.warps.assign(config_.core_warps, WarpDivergence{full_mask(), 0, {}});
        divergence_.poke(std::move(divergence));
        divergence_stack_.assign(static_cast<std::size_t>(config_.core_warps) * config_.core_divergence_stack_entries,
            DivergenceEntry{});
        depbar_.poke(DepbarCounters(launch.active_warps));
        shmem_to_ram_.poke(std::vector<uint32_t>(launch.active_warps, 0));
        sanitizer_.record_launch(config_, kernel_, launch.active_warps);
    } catch (const std::bad_alloc&) {
        throw SimError("machine state does not fit in host memory");
    } catch (const std::length_error&) {
        throw SimError("machine state exceeds host container limits");
    }
    trace_.emplace(
        launch.active_warps, config_.core_clock_hz, config_.dram_clock_hz, trace_options);
    trace_->set_mma_shapes(config_.mma_shapes);
    apply_image(shmem_, launch.test_shmem, "shmem");
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
        if (address % 8 != 0 || static_cast<uint64_t>(address) + 8 > shmem_.size()) {
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
//      (c) Issue looks ahead: an instruction accepted at t occupies its
//          resources from t+1, so issue sees the space this cycle frees
//          (queue entries, slots freed by this cycle's grants, the SIMT RMW
//          register emptied by an RMW start, the store buffer) and the
//          warp's shmem accesses as this cycle's grants leave them -- less
//          the space the dispatch pipeline's
//          instructions will take (dispatch_claims). Not the load slots
//          (an lds's or an atomic's return slot): issue counts the slots
//          empty in Q, so a slot whose RF write is granted at t is free for
//          a new admission from t + 1 (RF winner -> slot space -> winning
//          warp would be the longest path through issue). The RF write
//          decision's fairness inhibit, the LSU command slot its grant
//          releases, and
//          the acc.ld engine its grant lets advance are still seen at t.
//      (d) Same-cycle space reuse: a register drained at t may be refilled at
//          t (the load and store buffers, a load slot whose RF write is granted
//          at t -- by dispatch, which issue admitted earlier against the
//          slots' Q -- the TMA sector / store registers, the MMA event register,
//          an LSU slot or acc slot reallocated by dispatch). The refill's
//          data is visible at t+1; nothing passes through.
//      (e) A dispatched stg.v8 may make its first RF read beat in its
//          dispatch cycle.
//      (f) Inside the MMA: a result's tmem write precedes the same cycle's
//          accumulator read (write-first).
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
//          admission, A-read). A SIMT access or RMW accepted at issue sits in
//          its slot / register and is served from the next cycle. A new MMA
//          fetch serves its first port-B round in the cycle it starts.
//      (i) None from issue into fetch: the fetch decision reads registers
//          only (the
//          FIFOs' valid words, the read in flight, halted, the issue
//          register's branch). A slot an issue frees at t is seen by fetch
//          at t + 1; a branch accepted at t blocks its warp's fetch from
//          t + 1. Issue sees the FIFO heads' Q: a word landing at t issues
//          from t + 1.
// 2. DRIVERS. Each register's next value, from Q and the wires. They run in
//    any order (set_driver_shuffle_seed() permutes them every cycle).
// 3. CLOCK EDGE. Every register takes D; shmem takes the write rounds that
//    reached the RAM this cycle (A-write, then port-B write).
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

    // The RF write path, from Q and before any SIMT candidate: the one RF
    // write decision (the ready async holders share this cycle's unbooked
    // write ports; banks whose service is owed inhibit new SIMT bookings),
    // the write port (booked results and granted holders enter the write
    // pipeline) and its RAM stage (the writes landing at this cycle's edge,
    // and their completions).
    w.rf_write = decide_rf_writes();
    rf_write_port(w);
    rf_ram_stage(w);
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

    // Units whose next state issue looks ahead at (c). The MMA's commit may
    // present into the event register the RMW engine claimed just now (d).
    step_mma(w);
    step_acc_transfers(w);        // acc.ld advances on its RF grant; acc.st books its RF read before issue
    decide_mbar_timeouts(w);      // a wake this cycle wins over the deadline
    w.ingress = decide_ingress(w);
    step_gmem_controller(w);
    step_tma(w);

    // The TMA operand read's end: a tma's last words at the URF output now
    // push its command into the TMA queue. Then the dispatch pipeline's end:
    // a memory / engine / mbar instruction whose operands arrive now goes to
    // its unit.
    arrive_tma_operands(w);
    dispatch_issued(w);
    // The async RF readers (acc.st beats, the LSU's store-data read) take the
    // read ports the issue register's instruction leaves free now, from Q,
    // before issue.
    w.rf_read = decide_rf_reads(w);

    // The scheduler. Its write booking is in a future cycle and its read
    // round is the next cycle's (the issue register); it updates the owed
    // bits' next state.
    decide_issue(w);

    // The front ends' next state: issue's pop (and halt), the landing word,
    // the redirect, the fetch grant -- one calculation per warp.
    step_front_end(w);

    // The RF read stage: the issue register's round samples now (vector and
    // uniform registers). Then the URF write pipeline's work by stage: its
    // sampled write, the uniform ALU and r2ur's results, issue's new write.
    step_rf_reads(w);
    step_urf(w);

    run_drivers(w);

    // Clock edge: the registers, then the memories' writes at their RAM now.
    registers_.commit();
    for (const auto& write : w.shmem.writes) {
        std::memcpy(shmem_.data() + write.address, write.bytes.data(), write.bytes.size());
    }
    for (const auto& [word, value] : w.tmem_writes) tmem_[word] = value;
    if (w.branch.has_value() && w.branch->push.has_value()) divergence_stack_.at(w.branch->push_slot) = *w.branch->push;
    land_rf_writes(w);
    land_urf_write(w);
    ++cycle_;
    // Kernel end: the sanitizer's end-of-kernel checks (a nonzero mbar tx count).
    if (done()) sanitizer_.check_kernel_end(cycle_);
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
    }) && !fetch_read_.cur().has_value() && !divergence_.cur().refill.has_value();
    const auto& waits = mbar_wait_slots_.cur();
    const bool wait_slots_empty = std::all_of(
        waits.begin(), waits.end(), [](const auto& slot) { return !slot.has_value(); });
    const auto& tma = tma_.cur();
    const auto& mma = mma_.cur();
    const auto& acc = acc_engines_.cur();
    const auto empty = [](const auto& reg) { return !reg.cur().has_value(); };
    // The slots in every state, the probe register, the RMW
    // engine (its SIMT RMW register, operation and ALU stages) and the
    // services' pipelines (a store whose slot is free may still be travelling
    // to its write edge).
    const bool shmem_idle = std::all_of(load_slots_.begin(), load_slots_.end(), empty) &&
        std::all_of(store_slots_.begin(), store_slots_.end(), empty) && empty(mbar_probe_) && empty(simt_rmw_) &&
        !rmw_.cur().active.has_value() && rmw_alu_.empty() && a_read_pipe_.empty() &&
        a_write_pipe_.empty() && b_write_pipe_.empty() && empty(tma_read_txn_) &&
        empty(tma_write_txn_);
    return warps_halted && wait_slots_empty && shmem_idle && acc_ld_pipe_.empty() && acc_st_pipe_.empty() &&
        !tma_mbar_event_.cur().has_value() && !mma_mbar_event_.cur().has_value() && gmem_queue_.empty() &&
        !gmem_bus_.has_value() && pending_write_acks_.empty() && lsu_commands_.cur().empty() &&
        read_records_.cur().empty() && ingress_.cur().beats.empty() && egress_fifo_.cur().empty() &&
        !tma_sector_.cur().request.has_value() && tma_store_registers_.cur().held() == 0 &&
        tma.commands.empty() && tma.returns.empty() && !tma.event_command.has_value() &&
        tma.credits_used == 0 && tma_walker_start_pipe_.empty() && empty(urf_read_rounds_) &&
        tma_read_pairs_.empty() &&
        empty(tma_holding_) && writebacks_.cur().empty() && arithmetic_reads_.empty() &&
        dispatch_pipe_.empty() && lsu_read_pipe_.empty() && urf_pipe_.empty() &&
        mma.queue.empty() && mma_results_.empty() && !mma.fetch.has_value() && mma_reads_.empty() &&
        mma.arriving.empty() &&
        !acc[0].has_value() && !acc[1].has_value();
}

uint32_t Machine::read_shmem_u32(uint32_t byte_address) const {
    if (byte_address % 4 != 0 || static_cast<uint64_t>(byte_address) + 4 > shmem_.size()) {
        throw SimError("invalid host shmem u32 read");
    }
    uint32_t value = 0;
    std::memcpy(&value, shmem_.data() + byte_address, sizeof(value));
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
    if (byte_address % 8 != 0 || static_cast<uint64_t>(byte_address) + 8 > shmem_.size()) {
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

void Machine::test_set_fetch_pc(uint32_t warp, uint32_t pc) {
    auto front_end = front_end_.cur();
    front_end.at(warp).fetch_pc = pc;
    front_end_.poke(std::move(front_end));
}

uint32_t Machine::read_reg(
    uint32_t warp, uint32_t lane, uint32_t reg) const {
    if (warp >= warps_.size() || lane >= config_.lanes || reg >= kernel_.register_count) {
        throw SimError("invalid host RF read");
    }
    return rf(warps_[warp], lane, reg);
}

uint32_t Machine::read_ureg(uint32_t warp, uint32_t reg) const {
    if (warp >= warps_.size() || reg >= config_.urf_registers) throw SimError("invalid host URF read");
    return warps_[warp].urf[reg];
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
    static constexpr std::array<Driver, 41> canonical = {
        &Machine::drive_front_end,
        &Machine::drive_fetch,
        &Machine::drive_issue_branch,
        &Machine::drive_divergence,
        &Machine::drive_depbar,
        &Machine::drive_wait_slots,
        &Machine::drive_load_slots,
        &Machine::drive_store_slots,
        &Machine::drive_mbar_probe,
        &Machine::drive_simt_rmw,
        &Machine::drive_rmw_engine,
        &Machine::drive_a_read_pipe,
        &Machine::drive_write_pipes,
        &Machine::drive_shmem_arbiters,
        &Machine::drive_shmem_to_ram,
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
        &Machine::drive_tma_operand_read,
        &Machine::drive_lsu_commands,
        &Machine::drive_lsu_buffers,
        &Machine::drive_ingress,
        &Machine::drive_egress,
        &Machine::drive_read_records,
        &Machine::drive_free_id_tags,
        &Machine::drive_writebacks,
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
