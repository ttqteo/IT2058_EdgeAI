#!/bin/sh
# Build provenance-checked official TFLite Micro and the local host runner.
# It never resets or recursively cleans a vendor checkout.  GNU Make's
# genuine -B mode forces every upstream target to rebuild in place.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
VENDOR_DIR="$ROOT/artifacts/vendor"
TFLM_DIR="$VENDOR_DIR/tflite-micro"
TFLM_SHA=230c9d9531bd9b08d445b4f2ed0760756ffd5f78
TFLM_MAKEFILE="$TFLM_DIR/tensorflow/lite/micro/tools/make/Makefile"

MAKE_ROOT="$VENDOR_DIR/gnu-make-4.4.1"
MAKE_BIN="$MAKE_ROOT/install/bin/make"
MAKE_ARCHIVE="$MAKE_ROOT/make-4.4.1.tar.gz"
MAKE_URL="https://ftp.gnu.org/gnu/make/make-4.4.1.tar.gz"
MAKE_MD5=c8469a3713cbbe04d955d4ae4be23eeb
MAKE_SHA256=dd16fb1d67bfab79a72f5e8390735c49e3e8e70b4945a15ab1f81ddb78658fb3
RUNNER_MANIFEST="$ROOT/artifacts/tflm-host/edgeai_tflm_host.build.json"

fail() {
  echo "build_tflm_host: $*" >&2
  exit 1
}

hash_md5() {
  if command -v md5 >/dev/null 2>&1; then
    md5 -q "$1"
  elif command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | awk '{print $1}'
  else
    fail "neither md5 nor md5sum is available to verify GNU Make"
  fi
}

hash_sha256() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    fail "neither shasum nor sha256sum is available to record GNU Make SHA256"
  fi
}

verify_make_archive() {
  [ -f "$MAKE_ARCHIVE" ] || fail "GNU Make archive is missing: $MAKE_ARCHIVE"
  actual_md5=$(hash_md5 "$MAKE_ARCHIVE")
  [ "$actual_md5" = "$MAKE_MD5" ] || fail "GNU Make archive MD5 mismatch: $actual_md5"
  actual_sha256=$(hash_sha256 "$MAKE_ARCHIVE")
  [ "$actual_sha256" = "$MAKE_SHA256" ] || fail "GNU Make archive SHA256 mismatch: $actual_sha256"
  archive_size=$(wc -c < "$MAKE_ARCHIVE" | tr -d ' ')
  echo "GNU Make archive: bytes=$archive_size md5=$actual_md5 sha256=$actual_sha256"
}

make_version() {
  "$1" --version | sed -n '1s/^GNU Make //p'
}

make_is_supported() {
  version=$1
  [ -n "$version" ] || return 1
  major=${version%%.*}
  minor_part=${version#*.}
  minor=${minor_part%%.*}
  [ "$major" -gt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -ge 82 ]; }
}

ensure_gnu_make() {
  if [ -x "$MAKE_BIN" ]; then
    version=$(make_version "$MAKE_BIN")
    [ "$version" = "4.4.1" ] || fail "local Make binary is not GNU Make 4.4.1: $version"
    verify_make_archive
    echo "GNU Make binary: $MAKE_BIN (reported $version)"
    return
  fi

  system_make=$(command -v make || true)
  if [ -n "$system_make" ]; then
    system_version=$(make_version "$system_make" || true)
  else
    system_version=""
  fi
  if [ -n "$system_version" ] && make_is_supported "$system_version"; then
    echo "Using genuine system GNU Make $system_version: $system_make"
    MAKE_BIN=$system_make
    return
  fi

  mkdir -p "$MAKE_ROOT"
  if [ ! -f "$MAKE_ARCHIVE" ]; then
    command -v curl >/dev/null 2>&1 || fail "curl is required to download GNU Make 4.4.1"
    temporary_archive="$MAKE_ARCHIVE.tmp.$$"
    trap 'rm -f "$temporary_archive"' EXIT HUP INT TERM
    curl --fail --location --silent --show-error "$MAKE_URL" -o "$temporary_archive"
    mv "$temporary_archive" "$MAKE_ARCHIVE"
    trap - EXIT HUP INT TERM
  fi
  verify_make_archive

  source_dir="$MAKE_ROOT/make-4.4.1"
  if [ ! -d "$source_dir" ]; then
    tar -xzf "$MAKE_ARCHIVE" -C "$MAKE_ROOT"
  fi
  [ -f "$source_dir/configure" ] || fail "GNU Make source directory is incomplete: $source_dir"
  mkdir -p "$MAKE_ROOT/install"
  (
    cd "$source_dir"
    ./configure --prefix="$MAKE_ROOT/install"
    system_make=$(command -v make || true)
    [ -n "$system_make" ] || fail "a bootstrap make is required to build GNU Make"
    "$system_make" -j2
    "$system_make" install
  )
  [ -x "$MAKE_BIN" ] || fail "GNU Make 4.4.1 build did not produce $MAKE_BIN"
  version=$(make_version "$MAKE_BIN")
  [ "$version" = "4.4.1" ] || fail "built Make reports unexpected version: $version"
}

ensure_tflm_checkout() {
  mkdir -p "$VENDOR_DIR"
  if [ ! -d "$TFLM_DIR/.git" ]; then
    command -v git >/dev/null 2>&1 || fail "git is required to obtain pinned TFLM source"
    git clone --no-checkout https://github.com/tensorflow/tflite-micro.git "$TFLM_DIR"
    git -C "$TFLM_DIR" fetch --depth 1 origin "$TFLM_SHA"
    git -C "$TFLM_DIR" checkout --detach "$TFLM_SHA"
  fi
  [ -f "$TFLM_DIR/.git/HEAD" ] || fail "TFLM vendor path is not a git checkout: $TFLM_DIR"
  [ -z "$(git -C "$TFLM_DIR" status --porcelain --untracked-files=all)" ] || \
    fail "TFLM checkout is dirty; refusing to reset or overwrite it"
  actual_sha=$(git -C "$TFLM_DIR" rev-parse HEAD)
  [ "$actual_sha" = "$TFLM_SHA" ] || fail "TFLM checkout SHA mismatch: $actual_sha"
  [ -f "$TFLM_MAKEFILE" ] || fail "pinned TFLM Makefile is missing"
  echo "TFLM source: $actual_sha"
}

ensure_python_path() {
  [ -x "$ROOT/.venv/bin/python" ] || fail "repo .venv/bin/python is missing"
  [ -x "$ROOT/.venv/bin/python3" ] || fail "repo .venv/bin/python3 is missing"
  PATH="$ROOT/.venv/bin:$PATH"
  export PATH
  "$ROOT/.venv/bin/python" -c 'import absl, flatbuffers'
}

verify_tflm_checkout() {
  [ -z "$(git -C "$TFLM_DIR" status --porcelain --untracked-files=all)" ] || \
    fail "TFLM checkout became dirty during the build"
  actual_sha=$(git -C "$TFLM_DIR" rev-parse HEAD)
  [ "$actual_sha" = "$TFLM_SHA" ] || fail "TFLM checkout changed during the build: $actual_sha"
}

ensure_gnu_make
ensure_tflm_checkout
ensure_python_path

echo "Running official TFLM host test: test_hello_world_test"
(
  cd "$TFLM_DIR"
  "$MAKE_BIN" -B -j2 -f "$TFLM_MAKEFILE" test_hello_world_test
)

echo "Running official TFLM host test: test_memory_arena_threshold_test"
(
  cd "$TFLM_DIR"
  "$MAKE_BIN" -B -j2 -f "$TFLM_MAKEFILE" test_memory_arena_threshold_test
)

echo "Building official TFLM library target: libtflm"
(
  cd "$TFLM_DIR"
  "$MAKE_BIN" -B -j2 -f "$TFLM_MAKEFILE" libtflm
)
verify_tflm_checkout

echo "Building local host runner with genuine Make: $MAKE_BIN"
"$MAKE_BIN" -C "$ROOT/firmware/tflm_host" -f Makefile clean
"$MAKE_BIN" -C "$ROOT/firmware/tflm_host" -f Makefile CXX=clang++

echo "Running persisted direct-runner semantic tests"
# Do not let a sidecar from an earlier source/build pair masquerade as this
# run's evidence.  These are the two known generated manifest targets.
rm -f "$RUNNER_MANIFEST" "$RUNNER_MANIFEST.tmp"
"$ROOT/.venv/bin/python" -m pytest -q "$ROOT/tests/test_tflm_runner.py"
verify_tflm_checkout

compiler_version=$(clang++ --version | sed -n '1p')
echo "Writing local build manifest after build and tests"
"$ROOT/.venv/bin/python" "$ROOT/scripts/write_tflm_manifest.py" \
  --runner "$ROOT/artifacts/tflm-host/edgeai_tflm_host" \
  --tflm-checkout "$TFLM_DIR" \
  --tflm-commit "$TFLM_SHA" \
  --tflm-library "$TFLM_DIR/gen/osx_arm64_default_gcc/lib/libtensorflow-microlite.a" \
  --runner-source "$ROOT/firmware/tflm_host/runner.cc" \
  --runner-makefile "$ROOT/firmware/tflm_host/Makefile" \
  --build-script "$ROOT/scripts/build_tflm_host.sh" \
  --compiler-version "$compiler_version" \
  --compile-flags "-std=c++17 -O2 -Wall -Wextra -Werror -pedantic -Wno-c99-extensions -Wno-unused-parameter"

echo "Checking the newly written local build manifest"
"$ROOT/.venv/bin/python" -m pytest -q "$ROOT/tests/test_tflm_runner.py" \
  -k test_build_manifest_binds_current_local_artifacts

echo "Host runner: $ROOT/artifacts/tflm-host/edgeai_tflm_host"
echo "Build manifest: $RUNNER_MANIFEST"
