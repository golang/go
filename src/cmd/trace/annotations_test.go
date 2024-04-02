// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js

package main

import (
	"testing"
	"time"
)

func TestOverlappingDuration(t *testing.T) {
	cases := []struct {
		start0, end0, start1, end1 int64
		want                       time.Duration
	}{
		{
			1, 10, 11, 20, 0,
		},
		{
			1, 10, 5, 20, 5 * time.Nanosecond,
		},
		{
			1, 10, 2, 8, 6 * time.Nanosecond,
		},
	}

	for _, tc := range cases {
		s0, e0, s1, e1 := tc.start0, tc.end0, tc.start1, tc.end1
		if got := overlappingDuration(s0, e0, s1, e1); got != tc.want {
			t.Errorf("overlappingDuration(%d, %d, %d, %d)=%v; want %v", s0, e0, s1, e1, got, tc.want)
		}
		if got := overlappingDuration(s1, e1, s0, e0); got != tc.want {
			t.Errorf("overlappingDuration(%d, %d, %d, %d)=%v; want %v", s1, e1, s0, e0, got, tc.want)
		}
	}
}
