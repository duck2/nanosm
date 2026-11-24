#!/usr/bin/env python3
"""Tests of scripts/blackboxes.py.

  --fixtures  on small trees in a temporary directory: an unsigned black box,
              signing it with the fingerprint printed, a change inside it, a
              change to a shared part it uses (in its file and in another),
              line endings, every kind of malformed input refused, what is
              not a definition accepted; and in real temporary git
              repositories: --staged judging the index, not the working tree,
              an index with no source files, an unmerged entry, and the commit
              hook itself (git hook run pre-commit)
  --tree      on this tree: exactly ten black boxes, read cleanly; their states
              are printed, not pinned (the human signs them); on a copy of
              it, edits to what the contracts depend on outside their own
              markers change exactly the fingerprints they should (and a new
              non-arithmetic opcode none); and the machine's own black boxes
              (its state report: NANOSM_LIBRARY, or build/) are the marked
              ones with their regions' sign-offs
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "blackboxes.py"
HOOK = HERE / "hooks" / "pre-commit"
ROOT = HERE.parent
COMMENT = "// sign-off: only the human changes this line"

failures = []


def check(condition, message):
    if not condition:
        failures.append(message)
        print("FAIL: " + message)


def run(root, *args, env=None):
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--root", str(root)] + list(args),
                            capture_output=True, text=True, env=env)
    return result.returncode, result.stdout, result.stderr


def rows(stdout):
    """The black-box table's rows: name -> (where, state, fingerprint)."""
    found = {}
    lines = stdout.split("\n")
    for at, line in enumerate(lines):
        if re.match(r"^  black box\s{2,}sign-off line", line):
            for row in lines[at + 1:]:
                if not row.strip():
                    break
                cells = re.split(r"\s{2,}", row.strip())
                found[cells[0]] = tuple(cells[1:])
            break
    return found


def write(root, files, newline="\n"):
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.replace("\n", newline).encode())


SHARED = """// The helpers' description is outside every region.
// BLACK BOX SHARED: helpers
int helper() { return 1; }
// END BLACK BOX SHARED: helpers

void declare() {
    // BLACK BOX: adder
    // BLACK BOX USES: helpers
    BlackBoxDeclaration adder("adder", SignOff::not_signed); %s
    adder.latency = 1;
    // END BLACK BOX: adder
}
""" % COMMENT

MEMORY = """// BLACK BOX: buffer
// BLACK BOX USES: widths
Memory<int> buffer_;
BlackBoxDeclaration buffer_black_box() const {
    return memory_blackbox(buffer_,
        SignOff::not_signed, %s
        "latency", "readers", "writers");
}
// END BLACK BOX: buffer
""" % COMMENT

# A shared part in a file of its own, used by a black box in another; and
# what uses the declaration type without making a black box.
WIDTHS = """// BLACK BOX SHARED: widths
inline unsigned width(unsigned n) { return n; }
// END BLACK BOX SHARED: widths
BlackBoxDeclaration lookup(const std::string& name);
const std::vector<BlackBoxDeclaration>& black_boxes() const;
inline constexpr SignOff SignOff::not_signed{};
"""


def fixtures():
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = base / "tree"
        files = {"src/unit.cpp": SHARED, "src/sub/memories.hpp": MEMORY, "src/widths.hpp": WIDTHS}
        write(root, files)

        status, out, err = run(root)
        found = rows(out)
        check(status == 0 and not err, "the listing of a well-formed tree exits 0 (%d: %s)" % (status, err))
        check(sorted(found) == ["adder", "buffer"], "two black boxes listed: %s" % sorted(found))
        check(found.get("adder", ("",))[0] == "src/unit.cpp:9", "adder's sign-off line: %s" % (found.get("adder"),))
        check(found.get("buffer", ("",))[0] == "src/sub/memories.hpp:6",
              "buffer's sign-off line: %s" % (found.get("buffer"),))
        check(all(row[1] == "not signed" for row in found.values()), "both not signed: %s" % found)
        check(all(re.match(r"^[0-9a-f]{12}$", row[2]) for row in found.values()), "12-hex fingerprints: %s" % found)
        check("To sign one" in out, "the listing says how to sign")
        adder = found["adder"][2]
        buffer = found["buffer"][2]

        status, out, err = run(root, "--check")
        check(status == 1, "--check fails while unsigned (%d)" % status)
        check("adder" in out and "buffer" in out and "not signed" in out, "--check lists the unsigned: " + out)

        # Signing: the printed fingerprint pasted in place of not_signed.
        files["src/unit.cpp"] = SHARED.replace("SignOff::not_signed", 'SignOff::signed_by_human("%s")' % adder)
        write(root, files)
        status, out, err = run(root)
        found = rows(out)
        check(found["adder"][1:] == ("signed", adder), "adder signed with its own fingerprint: %s" % (found["adder"],))
        status, out, err = run(root, "--check")
        check(status == 1 and "buffer" in out and "adder" not in out, "--check lists only buffer: " + out)
        files["src/sub/memories.hpp"] = MEMORY.replace("SignOff::not_signed", 'SignOff::signed_by_human("%s")' % buffer)
        write(root, files)
        status, out, err = run(root, "--check")
        check(status == 0, "--check passes when every black box is signed (%d: %s %s)" % (status, out, err))
        signed = dict(files)

        # One character inside a region.
        files["src/unit.cpp"] = signed["src/unit.cpp"].replace("adder.latency = 1;", "adder.latency = 2;")
        write(root, files)
        status, out, err = run(root)
        found = rows(out)
        now = found["adder"][2]
        check(now != adder and found["adder"][1] == "changed since signing (signed for %s, is now %s)" % (adder, now),
              "a change inside adder: %s" % (found["adder"],))
        check(found["buffer"][1] == "signed", "buffer unaffected: %s" % (found["buffer"],))
        status, out, err = run(root, "--check")
        check(status == 1 and "changed since signing" in out and "adder" in out, "--check fails on a change: " + out)

        # One character in the shared part adder uses.
        files["src/unit.cpp"] = signed["src/unit.cpp"].replace("return 1;", "return 2;")
        write(root, files)
        found = rows(run(root)[1])
        check(found["adder"][1].startswith("changed since signing"), "a shared part's change un-signs: %s" % found)

        # One character in a shared part in another file: only its user changes.
        files = dict(signed)
        files["src/widths.hpp"] = WIDTHS.replace("return n;", "return n + 1;")
        write(root, files)
        found = rows(run(root)[1])
        check(found["buffer"][1].startswith("changed since signing") and found["adder"][1] == "signed",
              "a shared part in another file un-signs its user only: %s" % found)
        files = dict(signed)
        write(root, files)

        # Outside every region: nothing changes.
        files["src/unit.cpp"] = signed["src/unit.cpp"].replace("is outside every region", "is outside all regions")
        write(root, files)
        found = rows(run(root)[1])
        check(found["adder"][1:] == ("signed", adder), "a change outside the regions: %s" % (found["adder"],))

        # Line endings only.
        write(root, signed, newline="\r\n")
        status, out, err = run(root)
        found = rows(out)
        check(found["adder"][1:] == ("signed", adder) and found["buffer"][1:] == ("signed", buffer),
              "CRLF line endings change nothing: %s" % found)
        mixed = dict(signed)
        mixed["src/unit.cpp"] = signed["src/unit.cpp"].replace("adder.latency = 1;\n", "adder.latency = 1;\r\n")
        write(root, mixed)
        found = rows(run(root)[1])
        check(found["adder"][1:] == ("signed", adder), "mixed line endings change nothing: %s" % (found["adder"],))

        # --staged outside any repository reads the working tree.
        env = dict(os.environ)
        env["GIT_CEILING_DIRECTORIES"] = str(base)
        status, out, err = run(root, "--check", "--staged", env=env)
        check(status == 0 and "reading the working tree" in err, "--staged outside a repository (%d: %s)" % (status, err))

        # Malformed input: refused, nothing listed.
        box = lambda name, body="": (
            "// BLACK BOX: %s\nBlackBoxDeclaration b(\"%s\", SignOff::not_signed); %s\n%s// END BLACK BOX: %s\n"
            % (name, name, COMMENT, body, name))
        cases = {
            "an unclosed region": ({"src/a.cpp": "// BLACK BOX: a\nSignOff::not_signed, %s\n" % COMMENT},
                                   "is opened and never closed"),
            "two regions with one name": ({"src/a.cpp": box("a"), "src/b.cpp": box("a")}, "is defined twice"),
            "a declaration outside the markers": (
                {"src/a.cpp": box("a") + 'BlackBoxDeclaration stray("stray", SignOff::not_signed);\n'},
                "outside any BLACK BOX markers"),
            "a memory_blackbox call outside the markers": (
                {"src/a.cpp": box("a") + "auto m = memory_blackbox(rf_, SignOff::not_signed, \"\", \"\", \"\");\n"},
                "outside any BLACK BOX markers"),
            "a temporary declaration outside the markers": (
                {"src/a.cpp": box("a") + 'black_boxes_.push_back(BlackBoxDeclaration("t", SignOff::not_signed));\n'},
                "outside any BLACK BOX markers"),
            "a copy-initialised declaration outside the markers": (
                {"src/a.cpp": box("a") + 'auto t = BlackBoxDeclaration(\n    "t", SignOff::signed_by_human("0123456789ab"));\n'},
                "outside any BLACK BOX markers"),
            "an emplace_back outside the markers": (
                {"src/a.cpp": box("a") + 'black_boxes_.emplace_back("t", SignOff::not_signed);\n'},
                "outside any BLACK BOX markers"),
            "memory_blackbox with template arguments outside the markers": (
                {"src/a.cpp": box("a") + "auto m = memory_blackbox<uint32_t>(rf_, sign_off, \"\", \"\", \"\");\n"},
                "outside any BLACK BOX markers"),
            "a sign-off comment outside the markers": (
                {"src/a.cpp": box("a") + "    sign_off, %s\n" % COMMENT}, "outside any BLACK BOX markers"),
            "a region with no sign-off line": ({"src/a.cpp": "// BLACK BOX: a\nint x;\n// END BLACK BOX: a\n"},
                                               "has no sign-off line"),
            "a region with two sign-off lines": (
                {"src/a.cpp": box("a", "SignOff::not_signed, %s\n" % COMMENT)}, "has 2 sign-off lines"),
            "a region closed under another name": (
                {"src/a.cpp": "// BLACK BOX: a\nSignOff::not_signed, %s\n// END BLACK BOX: b\n" % COMMENT},
                "this closes black box 'b'"),
            "nested regions": ({"src/a.cpp": "// BLACK BOX: a\n" + box("b") + "// END BLACK BOX: a\n"},
                               "markers do not nest"),
            "a shared part nobody uses": (
                {"src/a.cpp": box("a") + "// BLACK BOX SHARED: s\n// END BLACK BOX SHARED: s\n"}, "used by no black box"),
            "a shared part that does not exist": (
                {"src/a.cpp": box("a", "// BLACK BOX USES: s\n")}, "which is not defined"),
            "a sign-off in a shared part": (
                {"src/a.cpp": box("a", "// BLACK BOX USES: s\n") +
                 "// BLACK BOX SHARED: s\nSignOff::not_signed, %s\n// END BLACK BOX SHARED: s\n" % COMMENT},
                "a shared part has none of its own"),
            "a signature that is not a fingerprint": (
                {"src/a.cpp": box("a").replace("SignOff::not_signed", 'SignOff::signed_by_human("xyz")')},
                "is not a fingerprint"),
            "a sign-off line without its comment": (
                {"src/a.cpp": box("a").replace(COMMENT, "// signed")}, "lacks its comment"),
            "a misspelt marker": ({"src/a.cpp": box("a") + "//BLACK BOX: c\n"}, "looks like a black-box marker"),
        }
        for number, (what, (tree, expected)) in enumerate(cases.items()):
            bad = base / ("bad%d" % number)
            write(bad, tree)
            for args in ([], ["--check"]):
                status, out, err = run(bad, *args)
                check(status == 2 and expected in err and not out.strip(),
                      "%s %s: exit %d, stderr %r, stdout %r" % (what, " ".join(args), status, err, out))
        print("fixtures: %d malformed cases refused" % len(cases))

        # Not a definition: prototypes and containers of the declaration
        # type, and the unsigned value's own definition.
        good = base / "good"
        write(good, {"src/a.cpp": box("a"), "src/b.hpp": WIDTHS.split("// END BLACK BOX SHARED: widths\n")[1]})
        status, out, err = run(good, "--check")
        check(status == 1 and not err, "prototypes returning BlackBoxDeclaration are not definitions (%d: %s)"
              % (status, err))


def git(repository, *args, stdin=None):
    return subprocess.run(["git", "-C", str(repository)] + list(args), input=stdin, capture_output=True)


def repositories():
    """--staged and the hook in real temporary repositories (git runs only there)."""
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        env = dict(os.environ)
        env["GIT_CEILING_DIRECTORIES"] = str(base)
        for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
            env.pop(key, None)

        # Staged and unstaged versions differing in a region.
        repository = base / "repository"
        repository.mkdir()
        check(git(repository, "init", "-q").returncode == 0, "git init")
        unsigned = {"src/unit.cpp": SHARED, "src/sub/memories.hpp": MEMORY, "src/widths.hpp": WIDTHS}
        write(repository, unsigned)
        git(repository, "add", "src")
        found = rows(run(repository)[1])
        fingerprints = {name: row[2] for name, row in found.items()}
        signed = {
            "src/unit.cpp": SHARED.replace("SignOff::not_signed",
                                           'SignOff::signed_by_human("%s")' % fingerprints["adder"]),
            "src/sub/memories.hpp": MEMORY.replace("SignOff::not_signed",
                                                   'SignOff::signed_by_human("%s")' % fingerprints["buffer"]),
        }
        write(repository, signed)  # signed in the working tree, not staged
        status, out, err = run(repository, "--check", "--staged", env=env)
        check(status == 1 and "adder" in out and "buffer" in out and not err,
              "--staged judges the index (unsigned), not the working tree (%d: %s %s)" % (status, out, err))
        check(run(repository, "--check", env=env)[0] == 0, "the working tree itself is signed")
        hook = install_hook(repository)
        result = subprocess.run(hook, capture_output=True, text=True, env=env)
        check(result.returncode != 0 and "Commit refused: these black boxes are not signed off" in result.stderr
              and fingerprints["adder"] in result.stderr and "To sign one" in result.stderr,
              "the hook refuses an unsigned index (%d: %s)" % (result.returncode, result.stderr))
        git(repository, "add", "src")
        status, out, err = run(repository, "--check", "--staged", env=env)
        check(status == 0 and not err, "--staged passes once the signed files are staged (%d: %s %s)" % (status, out, err))
        result = subprocess.run(hook, capture_output=True, text=True, env=env)
        check(result.returncode == 0 and not result.stderr.strip(),
              "the hook accepts a signed index (%d: %s)" % (result.returncode, result.stderr))
        changed = dict(signed)
        changed["src/unit.cpp"] = signed["src/unit.cpp"].replace("adder.latency = 1;", "adder.latency = 2;")
        write(repository, changed)  # changed in the working tree only
        status, out, err = run(repository, "--staged", env=env)
        check(status == 0 and rows(out).get("adder", ("", ""))[1] == "signed",
              "--staged keeps the staged, signed adder: %s" % out)
        check(run(repository, "--check", env=env)[0] == 1, "the working tree's adder has changed since signing")
        git(repository, "add", "src")
        result = subprocess.run(hook, capture_output=True, text=True, env=env)
        check(result.returncode != 0 and "changed since signing" in result.stderr,
              "the hook refuses a staged change to a signed black box (%d: %s)" % (result.returncode, result.stderr))

        # An index with no source files: nothing to judge (the working tree's
        # unsigned black box is not what is committed).
        documents = base / "documents"
        documents.mkdir()
        git(documents, "init", "-q")
        write(documents, {"README.md": "notes\n", "src/unit.cpp": SHARED})
        git(documents, "add", "README.md")
        status, out, err = run(documents, "--check", "--staged", env=env)
        check(status == 0 and "(0)" in out and not err,
              "--staged with no source in the index stays empty (%d: %s %s)" % (status, out, err))

        # An unmerged source file in the index: refused.
        merging = base / "merging"
        merging.mkdir()
        git(merging, "init", "-q")
        write(merging, {"src/unit.cpp": SHARED})
        blob = git(merging, "hash-object", "-w", "src/unit.cpp").stdout.decode().strip()
        info = "".join("100644 %s %d\tsrc/unit.cpp\n" % (blob, stage) for stage in (1, 2, 3))
        check(git(merging, "update-index", "--index-info", stdin=info.encode()).returncode == 0, "an unmerged index")
        status, out, err = run(merging, "--check", "--staged", env=env)
        check(status == 2 and "unmerged" in err and not out.strip(),
              "--staged refuses an unmerged source file (%d: %s %s)" % (status, out, err))
        print("repositories: --staged and the hook checked")


def install_hook(repository):
    """The hook and the script at their places in `repository`; the command that runs the hook as git does."""
    (repository / "scripts" / "hooks").mkdir(parents=True)
    shutil.copy(SCRIPT, repository / "scripts" / "blackboxes.py")
    shutil.copy(HOOK, repository / "scripts" / "hooks" / "pre-commit")
    return ["git", "-C", str(repository), "-c", "core.hooksPath=scripts/hooks", "hook", "run", "pre-commit"]


def tree():
    started = time.monotonic()
    status, out, err = run(ROOT)
    elapsed = time.monotonic() - started
    found = rows(out)
    check(status == 0 and not err, "the tree's black boxes are read cleanly (exit %d): %s" % (status, err))
    check(len(found) == 10, "the tree has %d black boxes, not ten: %s" % (len(found), sorted(found)))
    for name, (where, state, fingerprint) in found.items():
        check(state in ("not signed", "signed") or state.startswith("changed since signing (signed for "),
              "black box %s has the state %r" % (name, state))
        print("  %-24s %-34s %-12s %s" % (name, where, fingerprint, state))
    status, out, err = run(ROOT, "--check")
    check(status in (0, 1) and not err, "--check on the tree (exit %d): %s" % (status, err))
    print("tree: %d black boxes, read in %.2f s" % (len(found), elapsed))
    check(elapsed < 5, "the script took %.2f s" % elapsed)

    # On a copy: an edit to what a contract depends on outside its own
    # markers changes exactly the fingerprints of the black boxes that use it.
    arithmetic = {"ALU", "FPU", "special-function unit"}
    memories = {"vector register file", "uniform register file", "shared memory", "tensor memory",
                "divergence stack", "instruction memory"}
    edits = [
        ("a URF read port more", [("src/timing.hpp", "inline constexpr uint32_t urf_read_ports = 2;",
                                   "inline constexpr uint32_t urf_read_ports = 3;")], {"uniform register file"}),
        ("a slower instruction memory", [("src/timing.hpp", "inline constexpr uint32_t instruction_read_cycles = 1;",
                                          "inline constexpr uint32_t instruction_read_cycles = 2;")],
         {"instruction memory"}),
        ("bits_for changed", [("src/machine/blackbox.hpp", "n <= 1 ? 0 :", "n <= 2 ? 0 :")], arithmetic | memories),
        ("MemoryShape's ports swapped", [("src/machine/memory.hpp",
                                          "    uint32_t read_ports = 1;\n    uint32_t write_ports = 1;\n",
                                          "    uint32_t write_ports = 1;\n    uint32_t read_ports = 1;\n")], memories),
        # The routing: only the arithmetic opcodes' lists are in fingerprints,
        # each in its own unit's.
        ("rcp moved from the special-function unit to the FPU", [
            ("src/isa.cpp", "    case Opcode::ex2:\n    case Opcode::rcp:\n        return true;",
             "    case Opcode::ex2:\n        return true;"),
            ("src/isa.cpp", "    case Opcode::cvt_pack:\n        return true;",
             "    case Opcode::cvt_pack:\n    case Opcode::rcp:\n        return true;")],
         {"FPU", "special-function unit"}),
        ("a new non-arithmetic opcode", [
            ("src/isa.hpp", "    fence_sh = 0x27,\n", "    fence_sh = 0x27,\n    fence_gl = 0x28,\n"),
            ("src/isa.cpp", "    case Opcode::mma_commit:\n    case Opcode::fence_sh:\n",
             "    case Opcode::mma_commit:\n    case Opcode::fence_sh:\n    case Opcode::fence_gl:\n")], set()),
    ]
    before = {name: row[2] for name, row in found.items()}
    with tempfile.TemporaryDirectory() as temporary:
        copy = Path(temporary) / "tree"
        shutil.copytree(ROOT / "src", copy / "src")
        for what, changes, expected in edits:
            originals = {}
            for relative, old, new in changes:
                path = copy / relative
                originals.setdefault(path, path.read_bytes())
                text = path.read_bytes().decode().replace("\r\n", "\n")
                check(text.count(old) == 1, "%s no longer holds %r once: update this test's edit" % (relative, old))
                path.write_bytes(text.replace(old, new).encode())
            status, out, err = run(copy)
            after = {name: row[2] for name, row in rows(out).items()}
            moved = {name for name in before if after.get(name) != before[name]}
            check(status == 0 and moved == expected,
                  "%s changes %s, not %s (exit %d %s)" % (what, sorted(moved), sorted(expected), status, err))
            print("  %-52s changes %s" % (what, ", ".join(sorted(moved)) or "no fingerprint"))
            for path, original in originals.items():
                path.write_bytes(original)
    print("tree: %d edits outside the markers change exactly the fingerprints that depend on them" % len(edits))

    # The machine's own black boxes (its state report) are the marked
    # regions, each with the sign-off its region holds: a declaration made
    # any other way (a moved-from one pushed again, one left out of the list)
    # shows here.
    report = subprocess.run([sys.executable, "-B", "-m", "nanosm", "--state-report", "kernels/vadd.launch.py"],
                            cwd=str(ROOT), capture_output=True, text=True)
    check(report.returncode == 0, "the state report did not run (%d): %s" % (report.returncode, report.stderr))
    machine = []
    section = False
    for line in report.stdout.split("\n"):
        if line.startswith("black boxes ("):
            section = True
        elif section and not line.strip():
            break
        elif section:
            match = re.match(r"^  (\S.*) \([^()]*\): (?:not signed off|signed off ([0-9a-f]{12}))$", line)
            if match:
                machine.append((match.group(1), match.group(2)))
    signed_for = {}
    for name, (where, state, fingerprint) in found.items():
        changed = re.match(r"^changed since signing \(signed for ([0-9a-f]{12}),", state)
        signed_for[name] = fingerprint if state == "signed" else changed.group(1) if changed else None
    check(sorted(machine, key=str) == sorted(signed_for.items(), key=str),
          "the machine's black boxes and sign-offs %s are not the marked regions' %s"
          % (sorted(machine, key=str), sorted(signed_for.items(), key=str)))
    print("tree: the machine declares the %d marked black boxes, with their regions' sign-offs" % len(machine))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fixtures", action="store_true")
    parser.add_argument("--tree", action="store_true")
    args = parser.parse_args()
    if not (args.fixtures or args.tree):
        parser.error("say --fixtures, --tree or both")
    if args.fixtures:
        fixtures()
        repositories()
    if args.tree:
        tree()
    if failures:
        print("%d failures" % len(failures))
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
