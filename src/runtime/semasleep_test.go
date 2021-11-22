// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9 && !windows && !js

package runtime_test

import (
	"io"
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
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("StdoutPipe: %v", err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("Failed to start command: %v", err)
	}
	doneCh := make(chan error, 1)
	go func() {
		doneCh <- cmd.Wait()
	}()

	// Wait for After1 to close its stdout so that we know the runtime's SIGIO
	// handler is registered.
	b, err := io.ReadAll(stdout)
	if len(b) > 0 {
		t.Logf("read from testprog stdout: %s", b)
	}
	if err != nil {
		t.Fatalf("error reading from testprog: %v", err)
	}

	// With the repro running, we can continuously send to it
	// a signal that the runtime considers non-terminal,
	// such as SIGIO, to spuriously wake up
	// pthread_cond_timedwait_relative_np.
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case now := <-ticker.C:
			if now.Sub(start) > 2*time.Second {
				t.Error("Program failed to return on time and has to be killed, issue #27520 still exists")
				cmd.Process.Signal(syscall.SIGKILL)
				<-doneCh
				return
			}

			// Send the pesky signal that toggles spinning
			// indefinitely if #27520 is not fixed.
			cmd.Process.Signal(syscall.SIGIO)

		case err := <-doneCh:
			if err != nil {
				t.Fatalf("The program returned but unfortunately with an error: %v", err)
			}
			if time.Since(start) < 1*time.Second {
				// The program was supposed to sleep for a full (monotonic) second;
				// it should not return before that has elapsed.
				t.Fatalf("The program stopped too quickly.")
			}
			return
		}
	}
}
