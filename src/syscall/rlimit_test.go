// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package syscall_test

import (
	"internal/asan"
	"internal/msan"
	"internal/testenv"
	"os"
	"runtime"
	"syscall"
	"testing"
)

func TestOpenFileLimit(t *testing.T) {
	// For open file count,
	// macOS sets the default soft limit to 256 and no hard limit.
	// CentOS and Fedora set the default soft limit to 1024,
	// with hard limits of 4096 and 524288, respectively.
	// Check that we can open 1200 files, which proves
	// that the rlimit is being raised appropriately on those systems.
	fileCount := 1200

	// OpenBSD has a default soft limit of 512 and hard limit of 1024.
	if runtime.GOOS == "openbsd" {
		fileCount = 768
	}

	var files []*os.File
	for i := 0; i < fileCount; i++ {
		f, err := os.Open("rlimit.go")
		if err != nil {
			t.Error(err)
			break
		}
		files = append(files, f)
	}

	for _, f := range files {
		f.Close()
	}
}

// TestRlimitNofileSanitizerCap verifies that under ASan/MSan, syscall.init
// raises a low soft RLIMIT_NOFILE above its initial value while capping it at
// 100,000 even when the hard limit is higher.
func TestRlimitNofileSanitizerCap(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		var lim syscall.Rlimit
		if err := syscall.Getrlimit(syscall.RLIMIT_NOFILE, &lim); err != nil {
			t.Fatalf("Getrlimit: %v", err)
		}
		if lim.Cur <= 1024 || lim.Cur > 100_000 {
			t.Fatalf("soft RLIMIT_NOFILE = %d, want in (1024, 100000]", lim.Cur)
		}
		return
	}

	if !asan.Enabled && !msan.Enabled {
		t.Skip("only applies under ASan/MSan")
	}
	testenv.MustHaveExec(t)

	var orig syscall.Rlimit
	if err := syscall.Getrlimit(syscall.RLIMIT_NOFILE, &orig); err != nil {
		t.Fatalf("Getrlimit: %v", err)
	}
	if orig.Max <= 100_000 {
		t.Skipf("hard RLIMIT_NOFILE (%d) <= 100000", orig.Max)
	}

	origCached := syscall.GetInternalOrigRlimitNofile()
	defer origCached.Store(origCached.Load())
	defer syscall.Setrlimit(syscall.RLIMIT_NOFILE, &orig)

	low := syscall.Rlimit{Cur: 1024, Max: orig.Max}
	if err := syscall.Setrlimit(syscall.RLIMIT_NOFILE, &low); err != nil {
		t.Fatalf("Setrlimit: %v", err)
	}

	cmd := testenv.Command(t, testenv.Executable(t), "-test.run=^TestRlimitNofileSanitizerCap$")
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("child failed: %v\n%s", err, out)
	}
}
