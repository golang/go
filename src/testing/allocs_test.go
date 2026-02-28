// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing_test

import "testing"

var global any

var allocsPerRunTests = []struct {
	name   string
	fn     func()
	allocs float64
}{
	{"alloc *byte", func() { global = new(*byte) }, 1},
	{"alloc complex128", func() { global = new(complex128) }, 1},
	{"alloc float64", func() { global = new(float64) }, 1},
	{"alloc int32", func() { global = new(int32) }, 1},
	{"alloc byte", func() { global = new(byte) }, 1},
}

func TestAllocsPerRun(t *testing.T) {
	for _, tt := range allocsPerRunTests {
		if allocs := testing.AllocsPerRun(100, tt.fn); allocs != tt.allocs {
			t.Errorf("AllocsPerRun(100, %s) = %v, want %v", tt.name, allocs, tt.allocs)
		}
	}
}
