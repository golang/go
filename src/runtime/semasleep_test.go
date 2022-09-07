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
	t.Parallel() // Waits for a program to sleep for 1s.

	exe, err := buildTestProg(t, "testprog")
	if err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command(exe, "After1")
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("StdoutPipe: %v", err)
	}
	beforeStart := time.Now()
	if err := cmd.Start(); err != nil {
		t.Fatalf("Failed to start command: %v", err)
	}

	waiting := false
	doneCh := make(chan error, 1)
	t.Cleanup(func() {
		cmd.Process.Kill()
		if waiting {
			<-doneCh
		} else {
			cmd.Wait()
		}
	})

	// Wait for After1 to close its stdout so that we know the runtime's SIGIO
	// handler is registered.
	b, err := io.ReadAll(stdout)
	if len(b) > 0 {
		t.Logf("read from testprog stdout: %s", b)
	}
	if err != nil {
		t.Fatalf("error reading from testprog: %v", err)
	}

	// Wait for child exit.
	//
	// Note that we must do this after waiting for the write/child end of
	// stdout to close. Wait closes the read/parent end of stdout, so
	// starting this goroutine prior to io.ReadAll introduces a race
	// condition where ReadAll may get fs.ErrClosed if the child exits too
	// quickly.
	waiting = true
	go func() {
		doneCh <- cmd.Wait()
		close(doneCh)
	}()

	// Wait for an arbitrary timeout longer than one second. The subprocess itself
	// attempts to sleep for one second, but if the machine running the test is
	// heavily loaded that subprocess may not schedule very quickly even if the
	// bug remains fixed. (This is fine, because if the bug really is unfixed we
	// can keep the process hung indefinitely, as long as we signal it often
	// enough.)
	timeout := 10 * time.Second

	// The subprocess begins sleeping for 1s after it writes to stdout, so measure
	// the timeout from here (not from when we started creating the process).
	// That should reduce noise from process startup overhead.
	ready := time.Now()

	// With the repro running, we can continuously send to it
	// a signal that the runtime considers non-terminal,
	// such as SIGIO, to spuriously wake up
	// pthread_cond_timedwait_relative_np.
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case now := <-ticker.C:
			if now.Sub(ready) > timeout {
				t.Error("Program failed to return on time and has to be killed, issue #27520 still exists")
				// Send SIGQUIT to get a goroutine dump.
				// Stop sending SIGIO so that the program can clean up and actually terminate.
				cmd.Process.Signal(syscall.SIGQUIT)
				return
			}

			// Send the pesky signal that toggles spinning
			// indefinitely if #27520 is not fixed.
			cmd.Process.Signal(syscall.SIGIO)

		case err := <-doneCh:
			if err != nil {
				t.Fatalf("The program returned but unfortunately with an error: %v", err)
			}
			if time.Since(beforeStart) < 1*time.Second {
				// The program was supposed to sleep for a full (monotonic) second;
				// it should not return before that has elapsed.
				t.Fatalf("The program stopped too quickly.")
			}
			return
		}
	}
}
