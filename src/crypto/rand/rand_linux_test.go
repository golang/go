// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand_test

import (
	"bytes"
	"crypto/rand/internal/seccomp"
	"internal/syscall/unix"
	"internal/testenv"
	"os"
	"runtime"
	"syscall"
	"testing"
)

func TestNoGetrandom(t *testing.T) {
	if os.Getenv("GO_GETRANDOM_DISABLED") == "1" {
		// We are running under seccomp, the rest of the test suite will take
		// care of actually testing the implementation, we check that getrandom
		// is actually disabled.
		_, err := unix.GetRandom(make([]byte, 16), 0)
		if err != syscall.ENOSYS {
			t.Errorf("GetRandom returned %v, want ENOSYS", err)
		} else {
			t.Log("GetRandom returned ENOSYS as expected")
		}
		return
	}

	if testing.Short() {
		t.Skip("skipping test in short mode")
	}
	testenv.MustHaveExec(t)
	testenv.MustHaveCGO(t)

	done := make(chan struct{})
	go func() {
		defer close(done)
		// Call LockOSThread in a new goroutine, where we will apply the seccomp
		// filter. We exit without unlocking the thread, so the thread will die
		// and won't be reused.
		runtime.LockOSThread()

		if err := seccomp.DisableGetrandom(); err != nil {
			t.Errorf("failed to disable getrandom: %v", err)
			return
		}

		cmd := testenv.Command(t, os.Args[0], "-test.v")
		cmd.Env = append(os.Environ(), "GO_GETRANDOM_DISABLED=1")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Errorf("subprocess failed: %v\n%s", err, out)
			return
		}

		if !bytes.Contains(out, []byte("GetRandom returned ENOSYS")) {
			t.Errorf("subprocess did not disable getrandom")
		}
		if !bytes.Contains(out, []byte("TestRead")) {
			t.Errorf("subprocess did not run TestRead")
		}
	}()
	<-done
}
