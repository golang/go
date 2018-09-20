// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//+build !nacl,!plan9,!windows,!js

package runtime_test

import (
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// Issue #27250. Spurious wakeups to pthread_cond_timedwait_relative_np
// shouldn't cause semasleep to retry with the same timeout which would
// cause indefinite spinning.
func TestSpuriousWakeupsNeverHangSemasleep(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	tempDir, err := ioutil.TempDir("", "issue-27250")
	if err != nil {
		t.Fatalf("Failed to create the temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	repro := `
    package main

    import "time"

    func main() {
        <-time.After(1 * time.Second)
    }
    `
	mainPath := filepath.Join(tempDir, "main.go")
	if err := ioutil.WriteFile(mainPath, []byte(repro), 0644); err != nil {
		t.Fatalf("Failed to create temp file for repro.go: %v", err)
	}
	binaryPath := filepath.Join(tempDir, "binary")

	// Build the binary so that we can send the signal to its PID.
	out, err := exec.Command(testenv.GoToolPath(t), "build", "-o", binaryPath, mainPath).CombinedOutput()
	if err != nil {
		t.Fatalf("Failed to compile the binary: err: %v\nOutput: %s\n", err, out)
	}
	if err := os.Chmod(binaryPath, 0755); err != nil {
		t.Fatalf("Failed to chmod binary: %v", err)
	}

	// Now run the binary.
	cmd := exec.Command(binaryPath)
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
			return
		}
	}
}
