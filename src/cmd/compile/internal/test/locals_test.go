// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"sync/atomic"
	"testing"
)

func locals() {
	var x int64
	var y int32
	var z int16
	var w int8
	sink64 = &x
	sink32 = &y
	sink16 = &z
	sink8 = &w
}

//go:noinline
func args(x int64, y int32, z int16, w int8) {
	sink64 = &x
	sink32 = &y
	sink16 = &z
	sink8 = &w

}

//go:noinline
func half(x int64, y int16) {
	var z int32
	var w int8
	sink64 = &x
	sink16 = &y
	sink32 = &z
	sink8 = &w
}

//go:noinline
func closure() func() {
	var x int64
	var y int32
	var z int16
	var w int8
	_, _, _, _ = x, y, z, w
	return func() {
		x = 1
		y = 2
		z = 3
		w = 4
	}
}

//go:noinline
func atomicFn() {
	var x int32
	var y int64
	var z int16
	var w int8
	sink32 = &x
	sink64 = &y
	sink16 = &z
	sink8 = &w
	atomic.StoreInt64(&y, 7)
}

var sink64 *int64
var sink32 *int32
var sink16 *int16
var sink8 *int8

func TestLocalAllocations(t *testing.T) {
	type test struct {
		name string
		f    func()
		want int
	}
	for _, tst := range []test{
		{"locals", locals, 1},
		{"args", func() { args(1, 2, 3, 4) }, 1},
		{"half", func() { half(1, 2) }, 1},
		{"closure", func() { _ = closure() }, 2},
		{"atomic", atomicFn, 1},
	} {
		allocs := testing.AllocsPerRun(100, tst.f)
		if allocs != float64(tst.want) {
			t.Errorf("test %s uses %v allocs, want %d", tst.name, allocs, tst.want)
		}
	}
}
