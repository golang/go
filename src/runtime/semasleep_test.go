// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9 && !windows && !js
// +build !plan9,!windows,!js

package runtime_test

import (
	"os/exec"
	"syscall"
	"testing"
	"time"
)

// Issue #27250. Spurious wakeups to pthread_cond_timedwait_relative_np
// shouldn't cause semasleep to retry with the same timeout which would
// cause indefinite spinning.
func TestSpuriousWakeupsNeverHangSemasleep(t *testing.T) {
	if *flagQuick {
		t.Skip("-quick")
	}

	exe, err := buildTestProg(t, "testprog")
	if err != nil {
		t.Fatal(err)
	}

	start := time.Now()
	cmd := exec.Command(exe, "After1")
	if err := cmd.Start(); err != nil {
		t.Fatalf("Failed to start command: %v", err)
	}
	doneCh := make(chan error, 1)
	go func() {
		doneCh <- cmd.Wait()
	}()

	// With the repro running, we can continuously send to it
	// a non-terminal signal such as SIGIO, to spuriously
	// wakeup pthread_cond_timedwait_relative_np.
	unfixedTimer := time.NewTimer(2 * time.Second)
	for {
		select {
		case <-time.After(200 * time.Millisecond):
			// Send the pesky signal that toggles spinning
			// indefinitely if #27520 is not fixed.
			cmd.Process.Signal(syscall.SIGIO)

		case <-unfixedTimer.C:
			t.Error("Program failed to return on time and has to be killed, issue #27520 still exists")
			cmd.Process.Signal(syscall.SIGKILL)
			return

		case err := <-doneCh:
			if err != nil {
				t.Fatalf("The program returned but unfortunately with an error: %v", err)
			}
			if time.Since(start) < 100*time.Millisecond {
				t.Fatalf("The program stopped too quickly.")
			}
			return
		}
	}
}
