// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build runtime_test_sched

package main

import (
	"fmt"
	"os"
	_ "unsafe" // for go:linkname
)

//go:linkname runSpinExitHandoff runtime.schedTestRunSpinExitHandoff
func runSpinExitHandoff() (paused, ran, observed bool)

func init() {
	register("SchedulerSpinExitHandoff", schedulerSpinExitHandoff)
}

func schedulerSpinExitHandoff() {
	for range 10 {
		paused, ran, observed := runSpinExitHandoff()
		if !paused {
			fmt.Fprintln(os.Stderr, "spinner never paused before relinquishing responsibility")
			os.Exit(1)
		}
		if !ran {
			fmt.Fprintln(os.Stderr, "no other P ran the goroutine queued during the spin exit")
			os.Exit(1)
		}
		if observed {
			fmt.Println("OK")
			return
		}
		// Background work may bring the spinner back through schedule
		// before its per-P recheck. The queued G ran, but this execution
		// did not exercise the transition under test.
	}
	fmt.Fprintln(os.Stderr, "could not isolate the per-P queue recheck")
	os.Exit(1)
}
