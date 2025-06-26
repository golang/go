// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux || (freebsd && amd64)

package sanitizers_test

import (
	"bytes"
	"fmt"
	"internal/platform"
	"internal/testenv"
	"os/exec"
	"strings"
	"testing"
)

func TestASAN(t *testing.T) {
	config := mustHaveASAN(t)

	t.Parallel()
	mustRun(t, config.goCmd("build", "std"))

	cases := []struct {
		src               string
		memoryAccessError string
		errorLocation     string
		experiments       []string
	}{
		{src: "asan1_fail.go", memoryAccessError: "heap-use-after-free", errorLocation: "asan1_fail.go:25"},
		{src: "asan2_fail.go", memoryAccessError: "heap-buffer-overflow", errorLocation: "asan2_fail.go:31"},
		{src: "asan3_fail.go", memoryAccessError: "use-after-poison", errorLocation: "asan3_fail.go:13"},
		{src: "asan4_fail.go", memoryAccessError: "use-after-poison", errorLocation: "asan4_fail.go:13"},
		{src: "asan5_fail.go", memoryAccessError: "use-after-poison", errorLocation: "asan5_fail.go:18"},
		{src: "asan_useAfterReturn.go"},
		{src: "asan_unsafe_fail1.go", memoryAccessError: "use-after-poison", errorLocation: "asan_unsafe_fail1.go:25"},
		{src: "asan_unsafe_fail2.go", memoryAccessError: "use-after-poison", errorLocation: "asan_unsafe_fail2.go:25"},
		{src: "asan_unsafe_fail3.go", memoryAccessError: "use-after-poison", errorLocation: "asan_unsafe_fail3.go:18"},
		{src: "asan_global1_fail.go", memoryAccessError: "global-buffer-overflow", errorLocation: "asan_global1_fail.go:12"},
		{src: "asan_global2_fail.go", memoryAccessError: "global-buffer-overflow", errorLocation: "asan_global2_fail.go:19"},
		{src: "asan_global3_fail.go", memoryAccessError: "global-buffer-overflow", errorLocation: "asan_global3_fail.go:13"},
		{src: "asan_global4_fail.go", memoryAccessError: "global-buffer-overflow", errorLocation: "asan_global4_fail.go:21"},
		{src: "asan_global5.go"},
		{src: "asan_global_asm"},
		{src: "asan_global_asm2_fail", memoryAccessError: "global-buffer-overflow", errorLocation: "main.go:17"},
		{src: "arena_fail.go", memoryAccessError: "use-after-poison", errorLocation: "arena_fail.go:26", experiments: []string{"arenas"}},
	}
	for _, tc := range cases {
		tc := tc
		name := strings.TrimSuffix(tc.src, ".go")
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			dir := newTempDir(t)
			defer dir.RemoveAll(t)

			outPath := dir.Join(name)
			mustRun(t, config.goCmdWithExperiments("build", []string{"-o", outPath, srcPath(tc.src)}, tc.experiments))

			cmd := hangProneCmd(outPath)
			if tc.memoryAccessError != "" {
				outb, err := cmd.CombinedOutput()
				out := string(outb)
				if err != nil && strings.Contains(out, tc.memoryAccessError) {
					// This string is output if the
					// sanitizer library needs a
					// symbolizer program and can't find it.
					const noSymbolizer = "external symbolizer"
					// Check if -asan option can correctly print where the error occurred.
					if tc.errorLocation != "" &&
						!strings.Contains(out, tc.errorLocation) &&
						!strings.Contains(out, noSymbolizer) &&
						compilerSupportsLocation() {

						t.Errorf("%#q exited without expected location of the error\n%s; got failure\n%s", strings.Join(cmd.Args, " "), tc.errorLocation, out)
					}
					return
				}
				t.Fatalf("%#q exited without expected memory access error\n%s; got failure\n%s", strings.Join(cmd.Args, " "), tc.memoryAccessError, out)
			}
			mustRun(t, cmd)
		})
	}
}

func TestASANLinkerX(t *testing.T) {
	// Test ASAN with linker's -X flag (see issue 56175).
	config := mustHaveASAN(t)

	t.Parallel()

	dir := newTempDir(t)
	defer dir.RemoveAll(t)

	var ldflags string
	for i := 1; i <= 10; i++ {
		ldflags += fmt.Sprintf("-X=main.S%d=%d -X=cmd/cgo/internal/testsanitizers/testdata/asan_linkerx/p.S%d=%d ", i, i, i, i)
	}

	// build the binary
	outPath := dir.Join("main.exe")
	cmd := config.goCmd("build", "-ldflags="+ldflags, "-o", outPath)
	cmd.Dir = srcPath("asan_linkerx")
	mustRun(t, cmd)

	// run the binary
	mustRun(t, hangProneCmd(outPath))
}

// Issue 66966.
func TestASANFuzz(t *testing.T) {
	config := mustHaveASAN(t)

	t.Parallel()

	dir := newTempDir(t)
	defer dir.RemoveAll(t)

	exe := dir.Join("asan_fuzz_test.exe")
	cmd := config.goCmd("test", "-c", "-o", exe, srcPath("asan_fuzz_test.go"))
	t.Logf("%v", cmd)
	out, err := cmd.CombinedOutput()
	t.Logf("%s", out)
	if err != nil {
		t.Fatal(err)
	}

	cmd = exec.Command(exe, "-test.fuzz=Fuzz", "-test.fuzzcachedir="+dir.Base())
	cmd.Dir = dir.Base()
	t.Logf("%v", cmd)
	out, err = cmd.CombinedOutput()
	t.Logf("%s", out)
	if err == nil {
		t.Error("expected fuzzing failure")
	}
	if bytes.Contains(out, []byte("AddressSanitizer")) {
		t.Error(`output contains "AddressSanitizer", but should not`)
	}
}

func mustHaveASAN(t *testing.T) *config {
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	goos, err := goEnv("GOOS")
	if err != nil {
		t.Fatal(err)
	}
	goarch, err := goEnv("GOARCH")
	if err != nil {
		t.Fatal(err)
	}
	if !platform.ASanSupported(goos, goarch) {
		t.Skipf("skipping on %s/%s; -asan option is not supported.", goos, goarch)
	}

	// The current implementation is only compatible with the ASan library from version
	// v7 to v9 (See the description in src/runtime/asan/asan.go). Therefore, using the
	// -asan option must use a compatible version of ASan library, which requires that
	// the gcc version is not less than 7 and the clang version is not less than 9,
	// otherwise a segmentation fault will occur.
	if !compilerRequiredAsanVersion(goos, goarch) {
		t.Skipf("skipping on %s/%s: too old version of compiler", goos, goarch)
	}

	requireOvercommit(t)

	config := configure("address")
	config.skipIfCSanitizerBroken(t)

	return config
}
