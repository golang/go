// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd windows

package net

import (
	"testing"
	"time"
)

var deadlineSetTimeTests = []struct {
	input    time.Time
	expected int64
}{
	{time.Time{}, 0},
	{time.Date(2009, 11, 10, 23, 00, 00, 00, time.UTC), 1257894000000000000}, // 2009-11-10 23:00:00 +0000 UTC
}

func TestDeadlineSetTime(t *testing.T) {
	for _, tt := range deadlineSetTimeTests {
		var d deadline
		d.setTime(tt.input)
		actual := d.value()
		expected := int64(0)
		if !tt.input.IsZero() {
			expected = tt.input.UnixNano()
		}
		if actual != expected {
			t.Errorf("set/value failed: expected %v, actual %v", expected, actual)
		}
	}
}

var deadlineExpiredTests = []struct {
	deadline time.Time
	expired  bool
}{
	// note, times are relative to the start of the test run, not
	// the start of TestDeadlineExpired
	{time.Now().Add(5 * time.Minute), false},
	{time.Now().Add(-5 * time.Minute), true},
	{time.Time{}, false}, // no deadline set
}

func TestDeadlineExpired(t *testing.T) {
	for _, tt := range deadlineExpiredTests {
		var d deadline
		d.set(tt.deadline.UnixNano())
		expired := d.expired()
		if expired != tt.expired {
			t.Errorf("expire failed: expected %v, actual %v", tt.expired, expired)
		}
	}
}
