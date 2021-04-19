// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "testing"

// For GO386=387, make sure fucomi* opcodes are not used
// for comparison operations.
// Note that this test will fail only on a Pentium MMX
// processor (with GOARCH=386 GO386=387), as it just runs
// some code and looks for an unimplemented instruction fault.

//go:noinline
func compare1(a, b float64) bool {
	return a < b
}

//go:noinline
func compare2(a, b float32) bool {
	return a < b
}

func TestFloatCompare(t *testing.T) {
	if !compare1(3, 5) {
		t.Errorf("compare1 returned false")
	}
	if !compare2(3, 5) {
		t.Errorf("compare2 returned false")
	}
}

// For GO386=387, make sure fucomi* opcodes are not used
// for float->int conversions.

//go:noinline
func cvt1(a float64) uint64 {
	return uint64(a)
}

//go:noinline
func cvt2(a float64) uint32 {
	return uint32(a)
}

//go:noinline
func cvt3(a float32) uint64 {
	return uint64(a)
}

//go:noinline
func cvt4(a float32) uint32 {
	return uint32(a)
}

//go:noinline
func cvt5(a float64) int64 {
	return int64(a)
}

//go:noinline
func cvt6(a float64) int32 {
	return int32(a)
}

//go:noinline
func cvt7(a float32) int64 {
	return int64(a)
}

//go:noinline
func cvt8(a float32) int32 {
	return int32(a)
}

// make sure to cover int, uint cases (issue #16738)
//go:noinline
func cvt9(a float64) int {
	return int(a)
}

//go:noinline
func cvt10(a float64) uint {
	return uint(a)
}

//go:noinline
func cvt11(a float32) int {
	return int(a)
}

//go:noinline
func cvt12(a float32) uint {
	return uint(a)
}

func TestFloatConvert(t *testing.T) {
	if got := cvt1(3.5); got != 3 {
		t.Errorf("cvt1 got %d, wanted 3", got)
	}
	if got := cvt2(3.5); got != 3 {
		t.Errorf("cvt2 got %d, wanted 3", got)
	}
	if got := cvt3(3.5); got != 3 {
		t.Errorf("cvt3 got %d, wanted 3", got)
	}
	if got := cvt4(3.5); got != 3 {
		t.Errorf("cvt4 got %d, wanted 3", got)
	}
	if got := cvt5(3.5); got != 3 {
		t.Errorf("cvt5 got %d, wanted 3", got)
	}
	if got := cvt6(3.5); got != 3 {
		t.Errorf("cvt6 got %d, wanted 3", got)
	}
	if got := cvt7(3.5); got != 3 {
		t.Errorf("cvt7 got %d, wanted 3", got)
	}
	if got := cvt8(3.5); got != 3 {
		t.Errorf("cvt8 got %d, wanted 3", got)
	}
	if got := cvt9(3.5); got != 3 {
		t.Errorf("cvt9 got %d, wanted 3", got)
	}
	if got := cvt10(3.5); got != 3 {
		t.Errorf("cvt10 got %d, wanted 3", got)
	}
	if got := cvt11(3.5); got != 3 {
		t.Errorf("cvt11 got %d, wanted 3", got)
	}
	if got := cvt12(3.5); got != 3 {
		t.Errorf("cvt12 got %d, wanted 3", got)
	}
}
