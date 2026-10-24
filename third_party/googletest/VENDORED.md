# Vendored GoogleTest

GoogleTest v1.17.0, from
https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
(sha256 65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c).

Kept: the top-level CMakeLists.txt, LICENSE, README.md, and googletest/
(CMakeLists.txt, README.md, cmake/, include/, src/). Removed: googlemock/,
googletest/test/, googletest/samples/, docs/, ci/ and the Bazel files. The
build uses it with BUILD_GMOCK=OFF and INSTALL_GTEST=OFF (CMakeLists.txt),
so no network access is needed. To update: replace these files from a newer
release tag and update this note.
