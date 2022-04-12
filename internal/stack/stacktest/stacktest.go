// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stacktest

import (
	"testing"
	"time"

	"golang.org/x/tools/internal/stack"
)

// this is only needed to support pre 1.14 when testing.TB did not have Cleanup
type withCleanup interface {
	Cleanup(func())
}

// the maximum amount of time to wait for goroutines to clean themselves up.
const maxWait = time.Second

// NoLeak checks that a test (or benchmark) does not leak any goroutines.
func NoLeak(t testing.TB) {
	c, ok := t.(withCleanup)
	if !ok {
		return
	}
	before := stack.Capture()
	c.Cleanup(func() {
		var delta stack.Delta
		start := time.Now()
		delay := time.Millisecond
		for {
			after := stack.Capture()
			delta = stack.Diff(before, after)
			if len(delta.After) == 0 {
				// no leaks
				return
			}
			if time.Since(start) > maxWait {
				break
			}
			time.Sleep(delay)
			delay *= 2
		}
		// it's been long enough, and leaks are still present
		summary := stack.Summarize(delta.After)
		t.Errorf("goroutine leak detected:\n%+v", summary)
	})
}
