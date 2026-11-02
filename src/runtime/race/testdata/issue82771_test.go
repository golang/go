// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package race_test

import (
	"testing"
	"time"
)

// golang.org/issue/82771
// A race detected inside a channel timer's sendTime, run by the
// scheduler on g0, used to crash the process with SIGSEGV while
// symbolizing the report instead of reporting the race.
func TestRaceIssue82771(t *testing.T) {
	// The first tick must come after the write below, so that the race
	// is detected by the read in sendTime rather than by the write.
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	// A goroutine blocked on the channel keeps the timer in the heap,
	// so the scheduler runs sendTime when the P is otherwise idle.
	go func() {
		for range ticker.C {
		}
	}()

	time.Sleep(10 * time.Millisecond)
	// Unsynchronized write; sendTime calls time.Now, which reads time.Local.
	// Storing the current value keeps other tests unaffected.
	time.Local = issue82771Local()
	time.Sleep(100 * time.Millisecond)
}

// issue82771Local returns time.Local. It is not inlined so that the
// compiler cannot optimize away the store of time.Local to itself.
//
//go:noinline
func issue82771Local() *time.Location {
	return time.Local
}
