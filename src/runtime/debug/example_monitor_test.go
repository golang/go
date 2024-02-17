// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug_test

import (
	"io"
	"log"
	"os"
	"os/exec"
	"runtime/debug"
)

// ExampleSetCrashOutput_monitor shows an example of using
// [debug.SetCrashOutput] to direct crashes to a "monitor" process,
// for automated crash reporting. The monitor is the same executable,
// invoked in a special mode indicated by an environment variable.
func ExampleSetCrashOutput_monitor() {
	appmain()

	// This Example doesn't actually run as a test because its
	// purpose is to crash, so it has no "Output:" comment
	// within the function body.
	//
	// To observe the monitor in action, replace the entire text
	// of this comment with "Output:" and run this command:
	//
	//    $ go test -run=ExampleSetCrashOutput_monitor runtime/debug
	//    panic: oops
	//    ...stack...
	//    monitor: saved crash report at /tmp/10804884239807998216.crash
}

// appmain represents the 'main' function of your application.
func appmain() {
	monitor()

	// Run the application.
	println("hello")
	panic("oops")
}

// monitor starts the monitor process, which performs automated
// crash reporting. Call this function immediately within main.
//
// This function re-executes the same executable as a child process,
// in a special mode. In that mode, the call to monitor will never
// return.
func monitor() {
	const monitorVar = "RUNTIME_DEBUG_MONITOR"
	if os.Getenv(monitorVar) != "" {
		// This is the monitor (child) process.
		log.SetFlags(0)
		log.SetPrefix("monitor: ")

		crash, err := io.ReadAll(os.Stdin)
		if err != nil {
			log.Fatalf("failed to read from input pipe: %v", err)
		}
		if len(crash) == 0 {
			// Parent process terminated without reporting a crash.
			os.Exit(0)
		}

		// Save the crash report securely in the file system.
		f, err := os.CreateTemp("", "*.crash")
		if err != nil {
			log.Fatal(err)
		}
		if _, err := f.Write(crash); err != nil {
			log.Fatal(err)
		}
		if err := f.Close(); err != nil {
			log.Fatal(err)
		}
		log.Fatalf("saved crash report at %s", f.Name())
	}

	// This is the application process.
	// Fork+exec the same executable in monitor mode.
	exe, err := os.Executable()
	if err != nil {
		log.Fatal(err)
	}
	cmd := exec.Command(exe, "-test.run=ExampleSetCrashOutput_monitor")
	cmd.Env = append(os.Environ(), monitorVar+"=1")
	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stderr
	pipe, err := cmd.StdinPipe()
	if err != nil {
		log.Fatalf("StdinPipe: %v", err)
	}
	debug.SetCrashOutput(pipe.(*os.File)) // (this conversion is safe)
	if err := cmd.Start(); err != nil {
		log.Fatalf("can't start monitor: %v", err)
	}
	// Now return and start the application proper...
}
