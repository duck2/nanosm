#!/usr/bin/env python3
"""The black boxes and their sign-offs (src/machine/blackbox.hpp).

A black box is defined between two comment lines, `// BLACK BOX: <name>` and
`// END BLACK BOX: <name>`, with exactly one sign-off line between them:
`SignOff::not_signed` or `SignOff::signed_by_human("<fingerprint>")`, commented
`// sign-off: only the human changes this line`. Code that several black boxes
rely on is between `// BLACK BOX SHARED: <name>` and
`// END BLACK BOX SHARED: <name>`; a black box that relies on it says
`// BLACK BOX USES: <name>` between its own markers.

A black box's fingerprint: SHA-256 of its text (marker lines included, line
endings normalised to LF, the sign-off line's value replaced by a placeholder),
then of the text of each shared part it uses; its first 12 hex digits. The
human signs a black box by putting its fingerprint in its sign-off line; any
later change to that text gives another fingerprint, and the black box shows as
changed since signing.

Outside the markers nothing may make a black box: no SignOff:: (but the
definition of SignOff::not_signed itself), no sign-off comment, no use of
memory_blackbox.

  python scripts/blackboxes.py            every black box: where, its state, its fingerprint
  python scripts/blackboxes.py --check    exit 0 only if every black box is signed and unchanged
      --staged    read what is about to be committed (git's index; outside a repository, the
                  working tree, said on stderr), not the working tree
      --root DIR  the tree whose src/ is read (default: the directory above this script's)

Exit status: 0; 1 (--check: a black box not signed or changed since signing);
2 (the markers or git's index cannot be read: nothing is listed); 3 (the script failed).
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

SOURCE_SUFFIXES = {".cpp", ".hpp", ".h", ".cc", ".cxx", ".hh", ".hxx", ".inl", ".ipp"}
SIGN_OFF_COMMENT = "// sign-off: only the human changes this line"
PLACEHOLDER = "SignOff::<sign-off>"

OPEN = re.compile(r"^\s*// BLACK BOX: (\S.*?)\s*$")
CLOSE = re.compile(r"^\s*// END BLACK BOX: (\S.*?)\s*$")
SHARED_OPEN = re.compile(r"^\s*// BLACK BOX SHARED: (\S.*?)\s*$")
SHARED_CLOSE = re.compile(r"^\s*// END BLACK BOX SHARED: (\S.*?)\s*$")
USES = re.compile(r"^\s*// BLACK BOX USES: (\S.*?)\s*$")
LOOKS_LIKE_MARKER = re.compile(r"^\s*//\s*(END\s+)?BLACK\s*BOX")
SIGN_OFF_VALUE = re.compile(r'SignOff::(?:not_signed\b|signed_by_human\(\s*"([^"]*)"\s*\))')
# A black box cannot be made without a sign-off (BlackBoxDeclaration has no
# default, SignOff's constructors are private), so outside the markers any
# SignOff:: or sign-off comment is a black box defined there, and so is any
# use of memory_blackbox. The one exception: the unsigned value's own definition.
MEMORY_BLACKBOX = re.compile(r"\bmemory_blackbox\b")
NOT_SIGNED_DEFINITION = "inline constexpr SignOff SignOff::not_signed{};"
FINGERPRINT = re.compile(r"^[0-9a-f]{12}$")

BOX = "black box"
SHARED = "shared part"


class Malformed(Exception):
    pass


class Region:
    def __init__(self, kind, name, path, start):
        self.kind = kind
        self.name = name
        self.path = path
        self.start = start
        self.lines = []
        self.uses = []
        self.sign_offs = []  # (line number, index into lines)
        self.sign_off_line = None
        self.signed_for = None
        self.fingerprint = None


def code_of(line):
    """The line without its // comment."""
    return line.split("//", 1)[0]


def normalise(data):
    """A file's bytes as text with LF line endings."""
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode("utf-8", "surrogateescape")


def scan(path, text, errors):
    """The regions of one file; problems go to `errors` as (path, line, message)."""
    regions = []
    current = None
    for number, line in enumerate(text.split("\n"), 1):
        opened = [(kind, m) for kind, m in ((BOX, OPEN.match(line)), (SHARED, SHARED_OPEN.match(line))) if m]
        closed = [(kind, m) for kind, m in ((BOX, CLOSE.match(line)), (SHARED, SHARED_CLOSE.match(line))) if m]
        if opened:
            kind, match = opened[0]
            if current is not None:
                errors.append((path, number, "'%s' is opened inside %s '%s' (opened at line %d): markers do not nest"
                               % (match.group(1), current.kind, current.name, current.start)))
                continue
            current = Region(kind, match.group(1), path, number)
            current.lines.append(line)
            continue
        if closed:
            kind, match = closed[0]
            if current is None:
                errors.append((path, number, "'%s' is closed but was never opened" % match.group(1)))
            elif kind != current.kind or match.group(1) != current.name:
                errors.append((path, number, "this closes %s '%s', but %s '%s' is open (opened at line %d)"
                               % (kind, match.group(1), current.kind, current.name, current.start)))
                current = None
            else:
                current.lines.append(line)
                regions.append(current)
                current = None
            continue
        uses = USES.match(line)
        if not uses and LOOKS_LIKE_MARKER.match(line):
            errors.append((path, number, "this looks like a black-box marker but is not one: %s" % line.strip()))
            continue
        if current is None:
            code = code_of(line)
            if uses:
                errors.append((path, number, "'BLACK BOX USES' outside any black box's markers"))
            elif ("SignOff::" in code and code.strip() != NOT_SIGNED_DEFINITION) or SIGN_OFF_COMMENT in line \
                    or MEMORY_BLACKBOX.search(code):
                errors.append((path, number, "a black box is defined here outside any BLACK BOX markers "
                               "(a sign-off or memory_blackbox outside them)"))
            continue
        current.lines.append(line)
        if uses:
            if current.kind != BOX:
                errors.append((path, number, "'BLACK BOX USES' inside shared part '%s'" % current.name))
            elif uses.group(1) not in current.uses:
                current.uses.append(uses.group(1))
        if "SignOff::" in code_of(line) or SIGN_OFF_COMMENT in line:
            if current.kind != BOX:
                errors.append((path, number, "a sign-off in shared part '%s': a shared part has none of its own"
                               % current.name))
            else:
                current.sign_offs.append((number, len(current.lines) - 1))
    if current is not None:
        errors.append((path, current.start, "'%s' is opened and never closed" % current.name))
    for region in regions:
        if region.kind != BOX:
            continue
        if not region.sign_offs:
            errors.append((path, region.start, "black box '%s' has no sign-off line" % region.name))
            continue
        if len(region.sign_offs) > 1:
            errors.append((path, region.start, "black box '%s' has %d sign-off lines (lines %s)" % (
                region.name, len(region.sign_offs), ", ".join(str(n) for n, _ in region.sign_offs))))
            continue
        number, index = region.sign_offs[0]
        line = region.lines[index]
        values = list(SIGN_OFF_VALUE.finditer(code_of(line)))
        if len(values) != 1 or code_of(line).count("SignOff::") != 1:
            errors.append((path, number, "black box '%s': the sign-off line's value is neither SignOff::not_signed "
                           "nor SignOff::signed_by_human(\"<fingerprint>\")" % region.name))
            continue
        if SIGN_OFF_COMMENT not in line:
            errors.append((path, number, "black box '%s': the sign-off line lacks its comment '%s'"
                           % (region.name, SIGN_OFF_COMMENT)))
            continue
        signed_for = values[0].group(1)
        if signed_for is not None and not FINGERPRINT.match(signed_for):
            errors.append((path, number, "black box '%s': '%s' is not a fingerprint (12 hex digits, lower case)"
                           % (region.name, signed_for)))
            continue
        region.sign_off_line = number
        region.signed_for = signed_for
    return regions


def text_of(region):
    lines = list(region.lines)
    if region.kind == BOX:
        _, index = region.sign_offs[0]
        lines[index] = SIGN_OFF_VALUE.sub(PLACEHOLDER, lines[index], count=1)
    return "\n".join(lines) + "\n"


def read_black_boxes(files):
    """files: (relative path, bytes), in order. Returns (black boxes, shared parts) or raises Malformed."""
    errors = []
    regions = []
    for path, data in files:
        regions += scan(path, normalise(data), errors)
    boxes = [r for r in regions if r.kind == BOX]
    shared = {}
    seen = {}
    for region in regions:
        key = (region.kind, region.name)
        if key in seen:
            first = seen[key]
            errors.append((region.path, region.start, "%s '%s' is defined twice (also at %s:%d)"
                           % (region.kind, region.name, first.path, first.start)))
        else:
            seen[key] = region
            if region.kind == SHARED:
                shared[region.name] = region
    used = set()
    for box in boxes:
        for name in box.uses:
            used.add(name)
            if name not in shared:
                errors.append((box.path, box.start, "black box '%s' uses shared part '%s', which is not defined"
                               % (box.name, name)))
    for name, part in shared.items():
        if name not in used:
            errors.append((part.path, part.start, "shared part '%s' is used by no black box" % name))
    if errors:
        raise Malformed("\n".join("  %s:%d: %s" % error for error in errors))
    for box in boxes:
        digest = hashlib.sha256(text_of(box).encode("utf-8", "surrogateescape"))
        for name in box.uses:
            digest.update(b"\0")
            digest.update(text_of(shared[name]).encode("utf-8", "surrogateescape"))
        box.fingerprint = digest.hexdigest()[:12]
    return boxes, [shared[name] for name in shared]


def working_tree_files(root):
    found = []
    source = root / "src"
    if source.is_dir():
        for directory, _, names in os.walk(source):
            for name in names:
                path = Path(directory) / name
                if path.suffix in SOURCE_SUFFIXES:
                    found.append((path.relative_to(root).as_posix(), path))
    return [(relative, path.read_bytes()) for relative, path in sorted(found)]


def git(root, args, stdin=None):
    env = dict(os.environ)
    # A hook's git may name the index or the repository relative to where it runs.
    for key in ("GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE"):
        if env.get(key) and not os.path.isabs(env[key]):
            env[key] = os.path.abspath(env[key])
    return subprocess.run(["git"] + args, cwd=str(root), env=env, input=stdin, capture_output=True)


def staged_files(root):
    """What git's index holds under src/ (possibly nothing), or None (and why) outside a repository."""
    try:
        inside = git(root, ["rev-parse", "--is-inside-work-tree"])
    except OSError:
        return None, "git was not found"
    if inside.returncode != 0 or inside.stdout.strip() != b"true":
        return None, "not inside a git repository"
    listed = git(root, ["ls-files", "--stage", "-z", "--", "src"])
    if listed.returncode != 0:
        raise Malformed("  git ls-files failed: " + listed.stderr.decode(errors="replace").strip())
    entries = []
    unmerged = set()
    for record in listed.stdout.split(b"\0"):
        if not record:
            continue
        info, path = record.split(b"\t", 1)
        mode, sha, stage = info.split(b" ")
        relative = path.decode("utf-8", "surrogateescape")
        if not mode.startswith(b"100") or Path(relative).suffix not in SOURCE_SUFFIXES:
            continue
        if stage != b"0":
            unmerged.add(relative)
            continue
        entries.append((relative, sha))
    if unmerged:
        raise Malformed("\n".join("  %s: unmerged in git's index (finish the merge first)" % path
                                  for path in sorted(unmerged)))
    if not entries:
        return [], None
    blobs = git(root, ["cat-file", "--batch"], stdin=b"".join(sha + b"\n" for _, sha in entries))
    if blobs.returncode != 0:
        raise Malformed("  git cat-file failed: " + blobs.stderr.decode(errors="replace").strip())
    out = blobs.stdout
    files = []
    at = 0
    for relative, _ in entries:
        end = out.index(b"\n", at)
        size = int(out[at:end].split(b" ")[2])
        files.append((relative, out[end + 1:end + 1 + size]))
        at = end + 1 + size + 1
    return sorted(files), None


def state_of(box):
    if box.signed_for is None:
        return "not signed"
    if box.signed_for == box.fingerprint:
        return "signed"
    return "changed since signing (signed for %s, is now %s)" % (box.signed_for, box.fingerprint)


def table(rows):
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return ["  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows]


def main():
    parser = argparse.ArgumentParser(description="The black boxes and their sign-offs.")
    parser.add_argument("--check", action="store_true",
                        help="exit 0 only if every black box is signed and unchanged since signing")
    parser.add_argument("--staged", action="store_true",
                        help="read what is about to be committed (git's index), not the working tree")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="the tree whose src/ is read (default: %(default)s)")
    args = parser.parse_args()
    root = args.root.resolve()

    try:
        where = "the working tree"
        files = None
        if args.staged:
            files, why = staged_files(root)
            if files is None:
                print("blackboxes.py: --staged: %s; reading the working tree" % why, file=sys.stderr)
            else:
                where = "git's index (what is about to be committed)"
        if files is None:
            files = working_tree_files(root)
        boxes, shared = read_black_boxes(files)
    except Malformed as problem:
        print("blackboxes.py: the black boxes cannot be read:\n%s" % problem, file=sys.stderr)
        return 2

    location = {box.name: "%s:%d" % (box.path, box.sign_off_line) for box in boxes}
    if args.check:
        failing = [box for box in boxes if state_of(box) != "signed"]
        if not failing:
            print("every black box is signed off and unchanged since signing (%d)" % len(boxes))
            return 0
        for line in table([[box.name, location[box.name], state_of(box), box.fingerprint] for box in failing]):
            print(line)
        return 1

    states = [state_of(box) for box in boxes]
    signed = states.count("signed")
    unsigned = states.count("not signed")
    print("black boxes in src/, read from %s: %d -- %d signed, %d changed since signing, %d not signed"
          % (where, len(boxes), signed, len(boxes) - signed - unsigned, unsigned))
    if boxes:
        print()
        rows = [["black box", "sign-off line", "state", "fingerprint"]]
        rows += [[box.name, location[box.name], state, box.fingerprint] for box, state in zip(boxes, states)]
        for line in table(rows):
            print(line)
    if shared:
        print()
        rows = [["shared part", "where", "in the fingerprints of"]]
        for part in shared:
            users = [box.name for box in boxes if part.name in box.uses]
            rows.append([part.name, "%s:%d" % (part.path, part.start), ", ".join(users)])
        for line in table(rows):
            print(line)
    print()
    print('To sign one, write its fingerprint into its sign-off line: SignOff::signed_by_human("<fingerprint>") '
          "in place of the value there.")
    return 0


if __name__ == "__main__":
    # An error of the script itself is exit 3, never 1 (which says "not signed").
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(3)
