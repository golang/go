// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import "testing"

func TestPanicEvent(t *testing.T) {
	// Use a sync event for this because it doesn't have any extra metadata.
	ev := syncEvent(nil, 0)

	mustPanic(t, func() {
		_ = ev.Range()
	})
	mustPanic(t, func() {
		_ = ev.Metric()
	})
	mustPanic(t, func() {
		_ = ev.Log()
	})
	mustPanic(t, func() {
		_ = ev.Task()
	})
	mustPanic(t, func() {
		_ = ev.Region()
	})
	mustPanic(t, func() {
		_ = ev.Label()
	})
	mustPanic(t, func() {
		_ = ev.RangeAttributes()
	})
}

func mustPanic(t *testing.T, f func()) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("failed to panic")
		}
	}()
	f()
}
