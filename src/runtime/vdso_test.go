// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (freebsd && (386 || amd64 || arm || arm64 || riscv64)) || (linux && (386 || amd64 || arm || arm64 || loong64 || mips64 || mips64le || ppc64 || ppc64le || riscv64 || s390x))

package runtime_test

import (
	"bytes"
	"internal/asan"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// TestUsingVDSO tests that we are actually using the VDSO to fetch
// the time.
func TestUsingVDSO(t *testing.T) {
	if asan.Enabled {
		t.Skip("test fails with ASAN beause the ASAN leak checker won't run under strace")
	}

	const calls = 100

	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		// Fetch the time a lot.
		var total int64
		for i := 0; i < calls; i++ {
			total += time.Now().UnixNano()
		}
		os.Exit(0)
	}

	t.Parallel()

	// Look for strace in /bin or /usr/bin. Don't assume that some
	// strace on PATH is the one that we want.
	strace := "/bin/strace"
	if _, err := os.Stat(strace); err != nil {
		strace = "/usr/bin/strace"
		if _, err := os.Stat(strace); err != nil {
			t.Skipf("skipping test because strace not found: %v", err)
		}
	}

	exe, err := os.Executable()
	if err != nil {
		t.Skipf("skipping because Executable failed: %v", err)
	}

	t.Logf("GO_WANT_HELPER_PROCESS=1 %s -f -e clock_gettime %s -test.run=^TestUsingVDSO$", strace, exe)
	cmd := testenv.Command(t, strace, "-f", "-e", "clock_gettime", exe, "-test.run=^TestUsingVDSO$")
	cmd = testenv.CleanCmdEnv(cmd)
	cmd.Env = append(cmd.Env, "GO_WANT_HELPER_PROCESS=1")
	out, err := cmd.CombinedOutput()
	if len(out) > 0 {
		t.Logf("%s", out)
	}
	if err != nil {
		if err := err.(*exec.ExitError); err != nil && err.Sys().(syscall.WaitStatus).Signaled() {
			if !bytes.Contains(out, []byte("+++ killed by")) {
				// strace itself occasionally crashes.
				// Here, it exited with a signal, but
				// the strace log didn't report any
				// signal from the child process.
				t.Log(err)
				testenv.SkipFlaky(t, 63734)
			}
		}
		t.Fatal(err)
	}

	if got := bytes.Count(out, []byte("gettime")); got >= calls {
		t.Logf("found %d gettime calls, want < %d", got, calls)

		// Try to double-check that a C program uses the VDSO.
		tempdir := t.TempDir()
		cfn := filepath.Join(tempdir, "time.c")
		cexe := filepath.Join(tempdir, "time")
		if err := os.WriteFile(cfn, []byte(vdsoCProgram), 0o644); err != nil {
			t.Fatal(err)
		}
		cc := os.Getenv("CC")
		if cc == "" {
			cc, err = exec.LookPath("gcc")
			if err != nil {
				cc, err = exec.LookPath("clang")
				if err != nil {
					t.Skip("can't verify VDSO status, no C compiler")
				}
			}
		}

		t.Logf("%s -o %s %s", cc, cexe, cfn)
		cmd = testenv.Command(t, cc, "-o", cexe, cfn)
		cmd = testenv.CleanCmdEnv(cmd)
		out, err = cmd.CombinedOutput()
		if len(out) > 0 {
			t.Logf("%s", out)
		}
		if err != nil {
			t.Skipf("can't verify VDSO status, C compiled failed: %v", err)
		}

		t.Logf("%s -f -e clock_gettime %s", strace, cexe)
		cmd = testenv.Command(t, strace, "-f", "-e", "clock_gettime", cexe)
		cmd = testenv.CleanCmdEnv(cmd)
		out, err = cmd.CombinedOutput()
		if len(out) > 0 {
			t.Logf("%s", out)
		}
		if err != nil {
			t.Skipf("can't verify VDSO status, C program failed: %v", err)
		}

		if cgot := bytes.Count(out, []byte("gettime")); cgot >= 100 {
			t.Logf("found %d gettime calls, want < %d", cgot, 100)
			t.Log("C program does not use VDSO either")
			return
		}

		// The Go program used the system call but the C
		// program did not. This is a VDSO failure for Go.
		t.Errorf("did not use VDSO system call")
	}
}

const vdsoCProgram = `
#include <stdio.h>
#include <time.h>

int main() {
	int i;
	time_t tot;
	for (i = 0; i < 100; i++) {
		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		tot += ts.tv_nsec;
	}
	printf("%d\n", (int)(tot));
	return 0;
}
`
