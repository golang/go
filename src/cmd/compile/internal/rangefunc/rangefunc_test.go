// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.rangefunc

package rangefunc_test

import (
	"slices"
	"testing"
)

type Seq2[T1, T2 any] func(yield func(T1, T2) bool)

// OfSliceIndex returns a Seq over the elements of s. It is equivalent
// to range s.
func OfSliceIndex[T any, S ~[]T](s S) Seq2[int, T] {
	return func(yield func(int, T) bool) {
		for i, v := range s {
			if !yield(i, v) {
				return
			}
		}
		return
	}
}

// BadOfSliceIndex is "bad" because it ignores the return value from yield
// and just keeps on iterating.
func BadOfSliceIndex[T any, S ~[]T](s S) Seq2[int, T] {
	return func(yield func(int, T) bool) {
		for i, v := range s {
			yield(i, v)
		}
		return
	}
}

// VeryBadOfSliceIndex is "very bad" because it ignores the return value from yield
// and just keeps on iterating, and also wraps that call in a defer-recover so it can
// keep on trying after the first panic.
func VeryBadOfSliceIndex[T any, S ~[]T](s S) Seq2[int, T] {
	return func(yield func(int, T) bool) {
		for i, v := range s {
			func() {
				defer func() {
					recover()
				}()
				yield(i, v)
			}()
		}
		return
	}
}

// CooperativeBadOfSliceIndex calls the loop body from a goroutine after
// a ping on a channel, and returns recover()on that same channel.
func CooperativeBadOfSliceIndex[T any, S ~[]T](s S, proceed chan any) Seq2[int, T] {
	return func(yield func(int, T) bool) {
		for i, v := range s {
			if !yield(i, v) {
				// if the body breaks, call yield just once in a goroutine
				go func() {
					<-proceed
					defer func() {
						proceed <- recover()
					}()
					yield(0, s[0])
				}()
				return
			}
		}
		return
	}
}

// TrickyIterator is a type intended to test whether an iterator that
// calls a yield function after loop exit must inevitably escape the
// closure; this might be relevant to future checking/optimization.
type TrickyIterator struct {
	yield func(int, int) bool
}

func (ti *TrickyIterator) iterAll(s []int) Seq2[int, int] {
	return func(yield func(int, int) bool) {
		ti.yield = yield // Save yield for future abuse
		for i, v := range s {
			if !yield(i, v) {
				return
			}
		}
		return
	}
}

func (ti *TrickyIterator) iterOne(s []int) Seq2[int, int] {
	return func(yield func(int, int) bool) {
		ti.yield = yield // Save yield for future abuse
		if len(s) > 0 {  // Not in a loop might escape differently
			yield(0, s[0])
		}
		return
	}
}

func (ti *TrickyIterator) iterZero(s []int) Seq2[int, int] {
	return func(yield func(int, int) bool) {
		ti.yield = yield // Save yield for future abuse
		// Don't call it at all, maybe it won't escape
		return
	}
}

func (ti *TrickyIterator) fail() {
	if ti.yield != nil {
		ti.yield(1, 1)
	}
}

// Check wraps the function body passed to iterator forall
// in code that ensures that it cannot (successfully) be called
// either after body return false (control flow out of loop) or
// forall itself returns (the iteration is now done).
//
// Note that this can catch errors before the inserted checks.
func Check[U, V any](forall Seq2[U, V]) Seq2[U, V] {
	return func(body func(U, V) bool) {
		ret := true
		forall(func(u U, v V) bool {
			if !ret {
				panic("Checked iterator access after exit")
			}
			ret = body(u, v)
			return ret
		})
		ret = false
	}
}

func TestCheck(t *testing.T) {
	i := 0
	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw expected panic '%v'", r)
		} else {
			t.Error("Wanted to see a failure")
		}
	}()
	for _, x := range Check(BadOfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})) {
		i += x
		if i > 4*9 {
			break
		}
	}
}

func TestCooperativeBadOfSliceIndex(t *testing.T) {
	i := 0
	proceed := make(chan any)
	for _, x := range CooperativeBadOfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, proceed) {
		i += x
		if i >= 36 {
			break
		}
	}
	proceed <- true
	if r := <-proceed; r != nil {
		t.Logf("Saw expected panic '%v'", r)
	} else {
		t.Error("Wanted to see a failure")
	}
	if i != 36 {
		t.Errorf("Expected i == 36, saw %d instead", i)
	} else {
		t.Logf("i = %d", i)
	}
}

func TestCheckCooperativeBadOfSliceIndex(t *testing.T) {
	i := 0
	proceed := make(chan any)
	for _, x := range Check(CooperativeBadOfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, proceed)) {
		i += x
		if i >= 36 {
			break
		}
	}
	proceed <- true
	if r := <-proceed; r != nil {
		t.Logf("Saw expected panic '%v'", r)
	} else {
		t.Error("Wanted to see a failure")
	}
	if i != 36 {
		t.Errorf("Expected i == 36, saw %d instead", i)
	} else {
		t.Logf("i = %d", i)
	}
}

func TestTrickyIterAll(t *testing.T) {
	trickItAll := TrickyIterator{}
	i := 0
	for _, x := range trickItAll.iterAll([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
		i += x
		if i >= 36 {
			break
		}
	}

	if i != 36 {
		t.Errorf("Expected i == 36, saw %d instead", i)
	} else {
		t.Logf("i = %d", i)
	}

	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw expected panic '%v'", r)
		} else {
			t.Error("Wanted to see a failure")
		}
	}()

	trickItAll.fail()
}

func TestTrickyIterOne(t *testing.T) {
	trickItOne := TrickyIterator{}
	i := 0
	for _, x := range trickItOne.iterOne([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
		i += x
		if i >= 36 {
			break
		}
	}

	// Don't care about value, ought to be 36 anyhow.
	t.Logf("i = %d", i)

	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw expected panic '%v'", r)
		} else {
			t.Error("Wanted to see a failure")
		}
	}()

	trickItOne.fail()
}

func TestTrickyIterZero(t *testing.T) {
	trickItZero := TrickyIterator{}
	i := 0
	for _, x := range trickItZero.iterZero([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
		i += x
		if i >= 36 {
			break
		}
	}

	// Don't care about value, ought to be 0 anyhow.
	t.Logf("i = %d", i)

	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw expected panic '%v'", r)
		} else {
			t.Error("Wanted to see a failure")
		}
	}()

	trickItZero.fail()
}

func TestCheckTrickyIterZero(t *testing.T) {
	trickItZero := TrickyIterator{}
	i := 0
	for _, x := range Check(trickItZero.iterZero([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})) {
		i += x
		if i >= 36 {
			break
		}
	}

	// Don't care about value, ought to be 0 anyhow.
	t.Logf("i = %d", i)

	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw expected panic '%v'", r)
		} else {
			t.Error("Wanted to see a failure")
		}
	}()

	trickItZero.fail()
}

// TestBreak1 should just work, with well-behaved iterators.
// (The misbehaving iterator detector should not trigger.)
func TestBreak1(t *testing.T) {
	var result []int
	var expect = []int{1, 2, -1, 1, 2, -2, 1, 2, -3}
	for _, x := range OfSliceIndex([]int{-1, -2, -3, -4}) {
		if x == -4 {
			break
		}
		for _, y := range OfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
			if y == 3 {
				break
			}
			result = append(result, y)
		}
		result = append(result, x)
	}
	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
}

// TestBreak2 should just work, with well-behaved iterators.
// (The misbehaving iterator detector should not trigger.)
func TestBreak2(t *testing.T) {
	var result []int
	var expect = []int{1, 2, -1, 1, 2, -2, 1, 2, -3}
outer:
	for _, x := range OfSliceIndex([]int{-1, -2, -3, -4}) {
		for _, y := range OfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
			if y == 3 {
				break
			}
			if x == -4 {
				break outer
			}

			result = append(result, y)
		}
		result = append(result, x)
	}
	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
}

// TestContinue should just work, with well-behaved iterators.
// (The misbehaving iterator detector should not trigger.)
func TestContinue(t *testing.T) {
	var result []int
	var expect = []int{-1, 1, 2, -2, 1, 2, -3, 1, 2, -4}
outer:
	for _, x := range OfSliceIndex([]int{-1, -2, -3, -4}) {
		result = append(result, x)
		for _, y := range OfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
			if y == 3 {
				continue outer
			}
			if x == -4 {
				break outer
			}

			result = append(result, y)
		}
		result = append(result, x-10)
	}
	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
}

// TestBreak3 should just work, with well-behaved iterators.
// (The misbehaving iterator detector should not trigger.)
func TestBreak3(t *testing.T) {
	var result []int
	var expect = []int{100, 10, 2, 4, 200, 10, 2, 4, 20, 2, 4, 300, 10, 2, 4, 20, 2, 4, 30}
X:
	for _, x := range OfSliceIndex([]int{100, 200, 300, 400}) {
	Y:
		for _, y := range OfSliceIndex([]int{10, 20, 30, 40}) {
			if 10*y >= x {
				break
			}
			result = append(result, y)
			if y == 30 {
				continue X
			}
		Z:
			for _, z := range OfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
				if z&1 == 1 {
					continue Z
				}
				result = append(result, z)
				if z >= 4 {
					continue Y
				}
			}
			result = append(result, -y) // should never be executed
		}
		result = append(result, x)
	}
	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
}

// TestBreak1BadA should end in a panic when the outer-loop's
// single-level break is ignore by BadOfSliceIndex
func TestBreak1BadA(t *testing.T) {
	var result []int
	var expect = []int{1, 2, -1, 1, 2, -2, 1, 2, -3}

	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw expected panic '%v'", r)
			if !slices.Equal(expect, result) {
				t.Errorf("Expected %v, got %v", expect, result)
			}
		} else {
			t.Error("Wanted to see a failure")
		}
	}()

	for _, x := range BadOfSliceIndex([]int{-1, -2, -3, -4, -5}) {
		if x == -4 {
			break
		}
		for _, y := range OfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
			if y == 3 {
				break
			}
			result = append(result, y)
		}
		result = append(result, x)
	}
}

// TestBreak1BadB should end in a panic, sooner, when the inner-loop's
// (nested) single-level break is ignored by BadOfSliceIndex
func TestBreak1BadB(t *testing.T) {
	var result []int
	var expect = []int{1, 2} // inner breaks, panics, after before outer appends

	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw expected panic '%v'", r)
			if !slices.Equal(expect, result) {
				t.Errorf("Expected %v, got %v", expect, result)
			}
		} else {
			t.Error("Wanted to see a failure")
		}
	}()

	for _, x := range OfSliceIndex([]int{-1, -2, -3, -4, -5}) {
		if x == -4 {
			break
		}
		for _, y := range BadOfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
			if y == 3 {
				break
			}
			result = append(result, y)
		}
		result = append(result, x)
	}
}

// TestMultiCont0 tests multilevel continue with no bad iterators
// (it should just work)
func TestMultiCont0(t *testing.T) {
	var result []int
	var expect = []int{1000, 10, 2, 4, 2000}

W:
	for _, w := range OfSliceIndex([]int{1000, 2000}) {
		result = append(result, w)
		if w == 2000 {
			break
		}
		for _, x := range OfSliceIndex([]int{100, 200, 300, 400}) {
			for _, y := range OfSliceIndex([]int{10, 20, 30, 40}) {
				result = append(result, y)
				for _, z := range OfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
					if z&1 == 1 {
						continue
					}
					result = append(result, z)
					if z >= 4 {
						continue W // modified to be multilevel
					}
				}
				result = append(result, -y) // should never be executed
			}
			result = append(result, x)
		}
	}
	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
}

// TestMultiCont1 tests multilevel continue with a bad iterator
// in the outermost loop exited by the continue.
func TestMultiCont1(t *testing.T) {
	var result []int
	var expect = []int{1000, 10, 2, 4}
	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw expected panic '%v'", r)
			if !slices.Equal(expect, result) {
				t.Errorf("Expected %v, got %v", expect, result)
			}
		} else {
			t.Errorf("Wanted to see a failure, result was %v", result)
		}
	}()

W:
	for _, w := range OfSliceIndex([]int{1000, 2000}) {
		result = append(result, w)
		if w == 2000 {
			break
		}
		for _, x := range BadOfSliceIndex([]int{100, 200, 300, 400}) {
			for _, y := range OfSliceIndex([]int{10, 20, 30, 40}) {
				result = append(result, y)
				for _, z := range OfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
					if z&1 == 1 {
						continue
					}
					result = append(result, z)
					if z >= 4 {
						continue W
					}
				}
				result = append(result, -y) // should never be executed
			}
			result = append(result, x)
		}
	}
	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
}

// TestMultiCont2 tests multilevel continue with a bad iterator
// in a middle loop exited by the continue.
func TestMultiCont2(t *testing.T) {
	var result []int
	var expect = []int{1000, 10, 2, 4}
	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw expected panic '%v'", r)
			if !slices.Equal(expect, result) {
				t.Errorf("Expected %v, got %v", expect, result)
			}
		} else {
			t.Errorf("Wanted to see a failure, result was %v", result)
		}
	}()

W:
	for _, w := range OfSliceIndex([]int{1000, 2000}) {
		result = append(result, w)
		if w == 2000 {
			break
		}
		for _, x := range OfSliceIndex([]int{100, 200, 300, 400}) {
			for _, y := range BadOfSliceIndex([]int{10, 20, 30, 40}) {
				result = append(result, y)
				for _, z := range OfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
					if z&1 == 1 {
						continue
					}
					result = append(result, z)
					if z >= 4 {
						continue W
					}
				}
				result = append(result, -y) // should never be executed
			}
			result = append(result, x)
		}
	}
	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
}

// TestMultiCont3 tests multilevel continue with a bad iterator
// in the innermost loop exited by the continue.
func TestMultiCont3(t *testing.T) {
	var result []int
	var expect = []int{1000, 10, 2, 4}
	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw expected panic '%v'", r)
			if !slices.Equal(expect, result) {
				t.Errorf("Expected %v, got %v", expect, result)
			}
		} else {
			t.Errorf("Wanted to see a failure, result was %v", result)
		}
	}()

W:
	for _, w := range OfSliceIndex([]int{1000, 2000}) {
		result = append(result, w)
		if w == 2000 {
			break
		}
		for _, x := range OfSliceIndex([]int{100, 200, 300, 400}) {
			for _, y := range OfSliceIndex([]int{10, 20, 30, 40}) {
				result = append(result, y)
				for _, z := range BadOfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
					if z&1 == 1 {
						continue
					}
					result = append(result, z)
					if z >= 4 {
						continue W
					}
				}
				result = append(result, -y) // should never be executed
			}
			result = append(result, x)
		}
	}
	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
}

// TestMultiBreak0 tests multilevel break with a bad iterator
// in the outermost loop exited by the break (the outermost loop).
func TestMultiBreak0(t *testing.T) {
	var result []int
	var expect = []int{1000, 10, 2, 4}
	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw expected panic '%v'", r)
			if !slices.Equal(expect, result) {
				t.Errorf("Expected %v, got %v", expect, result)
			}
		} else {
			t.Errorf("Wanted to see a failure, result was %v", result)
		}
	}()

W:
	for _, w := range BadOfSliceIndex([]int{1000, 2000}) {
		result = append(result, w)
		if w == 2000 {
			break
		}
		for _, x := range OfSliceIndex([]int{100, 200, 300, 400}) {
			for _, y := range OfSliceIndex([]int{10, 20, 30, 40}) {
				result = append(result, y)
				for _, z := range OfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
					if z&1 == 1 {
						continue
					}
					result = append(result, z)
					if z >= 4 {
						break W
					}
				}
				result = append(result, -y) // should never be executed
			}
			result = append(result, x)
		}
	}
	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
}

// TestMultiBreak1 tests multilevel break with a bad iterator
// in an intermediate loop exited by the break.
func TestMultiBreak1(t *testing.T) {
	var result []int
	var expect = []int{1000, 10, 2, 4}
	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw expected panic '%v'", r)
			if !slices.Equal(expect, result) {
				t.Errorf("Expected %v, got %v", expect, result)
			}
		} else {
			t.Errorf("Wanted to see a failure, result was %v", result)
		}
	}()

W:
	for _, w := range OfSliceIndex([]int{1000, 2000}) {
		result = append(result, w)
		if w == 2000 {
			break
		}
		for _, x := range BadOfSliceIndex([]int{100, 200, 300, 400}) {
			for _, y := range OfSliceIndex([]int{10, 20, 30, 40}) {
				result = append(result, y)
				for _, z := range OfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
					if z&1 == 1 {
						continue
					}
					result = append(result, z)
					if z >= 4 {
						break W
					}
				}
				result = append(result, -y) // should never be executed
			}
			result = append(result, x)
		}
	}
	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
}

// TestMultiBreak2 tests multilevel break with two bad iterators
// in intermediate loops exited by the break.
func TestMultiBreak2(t *testing.T) {
	var result []int
	var expect = []int{1000, 10, 2, 4}
	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw expected panic '%v'", r)
			if !slices.Equal(expect, result) {
				t.Errorf("Expected %v, got %v", expect, result)
			}
		} else {
			t.Errorf("Wanted to see a failure, result was %v", result)
		}
	}()

W:
	for _, w := range OfSliceIndex([]int{1000, 2000}) {
		result = append(result, w)
		if w == 2000 {
			break
		}
		for _, x := range BadOfSliceIndex([]int{100, 200, 300, 400}) {
			for _, y := range BadOfSliceIndex([]int{10, 20, 30, 40}) {
				result = append(result, y)
				for _, z := range OfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
					if z&1 == 1 {
						continue
					}
					result = append(result, z)
					if z >= 4 {
						break W
					}
				}
				result = append(result, -y) // should never be executed
			}
			result = append(result, x)
		}
	}
	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
}

// TestMultiBreak3 tests multilevel break with the bad iterator
// in the innermost loop exited by the break.
func TestMultiBreak3(t *testing.T) {
	var result []int
	var expect = []int{1000, 10, 2, 4}
	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw expected panic '%v'", r)
			if !slices.Equal(expect, result) {
				t.Errorf("Expected %v, got %v", expect, result)
			}
		} else {
			t.Errorf("Wanted to see a failure, result was %v", result)
		}
	}()

W:
	for _, w := range OfSliceIndex([]int{1000, 2000}) {
		result = append(result, w)
		if w == 2000 {
			break
		}
		for _, x := range OfSliceIndex([]int{100, 200, 300, 400}) {
			for _, y := range OfSliceIndex([]int{10, 20, 30, 40}) {
				result = append(result, y)
				for _, z := range BadOfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
					if z&1 == 1 {
						continue
					}
					result = append(result, z)
					if z >= 4 {
						break W
					}
				}
				result = append(result, -y) // should never be executed
			}
			result = append(result, x)
		}
	}
	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
}

// veryBad tests that a loop nest behaves sensibly in the face of a
// "very bad" iterator.  In this case, "sensibly" means that the
// break out of X still occurs after the very bad iterator finally
// quits running (the control flow bread crumbs remain.)
func veryBad(s []int) []int {
	var result []int
X:
	for _, x := range OfSliceIndex([]int{1, 2, 3}) {

		result = append(result, x)

		for _, y := range VeryBadOfSliceIndex(s) {
			result = append(result, y)
			break X
		}
		for _, z := range OfSliceIndex([]int{100, 200, 300}) {
			result = append(result, z)
			if z == 100 {
				break
			}
		}
	}
	return result
}

// checkVeryBad wraps a "very bad" iterator with Check,
// demonstrating that the very bad iterator also hides panics
// thrown by Check.
func checkVeryBad(s []int) []int {
	var result []int
X:
	for _, x := range OfSliceIndex([]int{1, 2, 3}) {

		result = append(result, x)

		for _, y := range Check(VeryBadOfSliceIndex(s)) {
			result = append(result, y)
			break X
		}
		for _, z := range OfSliceIndex([]int{100, 200, 300}) {
			result = append(result, z)
			if z == 100 {
				break
			}
		}
	}
	return result
}

// okay is the not-bad version of veryBad.
// They should behave the same.
func okay(s []int) []int {
	var result []int
X:
	for _, x := range OfSliceIndex([]int{1, 2, 3}) {

		result = append(result, x)

		for _, y := range OfSliceIndex(s) {
			result = append(result, y)
			break X
		}
		for _, z := range OfSliceIndex([]int{100, 200, 300}) {
			result = append(result, z)
			if z == 100 {
				break
			}
		}
	}
	return result
}

// TestVeryBad1 checks the behavior of an extremely poorly behaved iterator.
func TestVeryBad1(t *testing.T) {
	result := veryBad([]int{10, 20, 30, 40, 50}) // odd length
	expect := []int{1, 10}

	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
}

// TestVeryBad2 checks the behavior of an extremely poorly behaved iterator.
func TestVeryBad2(t *testing.T) {
	result := veryBad([]int{10, 20, 30, 40}) // even length
	expect := []int{1, 10}

	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
}

// TestCheckVeryBad checks the behavior of an extremely poorly behaved iterator,
// which also suppresses the exceptions from "Check"
func TestCheckVeryBad(t *testing.T) {
	result := checkVeryBad([]int{10, 20, 30, 40}) // even length
	expect := []int{1, 10}

	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
}

// TestOk is the nice version of the very bad iterator.
func TestOk(t *testing.T) {
	result := okay([]int{10, 20, 30, 40, 50}) // odd length
	expect := []int{1, 10}

	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
}

// testBreak1BadDefer checks that defer behaves properly even in
// the presence of loop bodies panicking out of bad iterators.
// (i.e., the instrumentation did not break defer in these loops)
func testBreak1BadDefer(t *testing.T) (result []int) {
	var expect = []int{1, 2, -1, 1, 2, -2, 1, 2, -3, -30, -20, -10}

	defer func() {
		if r := recover(); r != nil {
			t.Logf("Saw expected panic '%v'", r)
			if !slices.Equal(expect, result) {
				t.Errorf("(Inner) Expected %v, got %v", expect, result)
			}
		} else {
			t.Error("Wanted to see a failure")
		}
	}()

	for _, x := range BadOfSliceIndex([]int{-1, -2, -3, -4, -5}) {
		if x == -4 {
			break
		}
		defer func() {
			result = append(result, x*10)
		}()
		for _, y := range OfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
			if y == 3 {
				break
			}
			result = append(result, y)
		}
		result = append(result, x)
	}
	return
}

func TestBreak1BadDefer(t *testing.T) {
	var result []int
	var expect = []int{1, 2, -1, 1, 2, -2, 1, 2, -3, -30, -20, -10}
	result = testBreak1BadDefer(t)
	if !slices.Equal(expect, result) {
		t.Errorf("(Outer) Expected %v, got %v", expect, result)
	}
}

// testReturn1 has no bad iterators.
func testReturn1(t *testing.T) (result []int, err any) {
	defer func() {
		err = recover()
	}()
	for _, x := range OfSliceIndex([]int{-1, -2, -3, -4, -5}) {
		result = append(result, x)
		if x == -4 {
			break
		}
		defer func() {
			result = append(result, x*10)
		}()
		for _, y := range OfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
			if y == 3 {
				return
			}
			result = append(result, y)
		}
		result = append(result, x)
	}
	return
}

// testReturn2 has an outermost bad iterator
func testReturn2(t *testing.T) (result []int, err any) {
	defer func() {
		err = recover()
	}()
	for _, x := range BadOfSliceIndex([]int{-1, -2, -3, -4, -5}) {
		result = append(result, x)
		if x == -4 {
			break
		}
		defer func() {
			result = append(result, x*10)
		}()
		for _, y := range OfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
			if y == 3 {
				return
			}
			result = append(result, y)
		}
		result = append(result, x)
	}
	return
}

// testReturn3 has an innermost bad iterator
func testReturn3(t *testing.T) (result []int, err any) {
	defer func() {
		err = recover()
	}()
	for _, x := range OfSliceIndex([]int{-1, -2, -3, -4, -5}) {
		result = append(result, x)
		if x == -4 {
			break
		}
		defer func() {
			result = append(result, x*10)
		}()
		for _, y := range BadOfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
			if y == 3 {
				return
			}
			result = append(result, y)
		}
	}
	return
}

// TestReturns checks that returns through bad iterators behave properly,
// for inner and outer bad iterators.
func TestReturns(t *testing.T) {
	var result []int
	var expect = []int{-1, 1, 2, -10}
	var err any

	result, err = testReturn1(t)
	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}

	result, err = testReturn2(t)
	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
	if err == nil {
		t.Errorf("Missing expected error")
	} else {
		t.Logf("Saw expected panic '%v'", err)
	}

	result, err = testReturn3(t)
	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
	if err == nil {
		t.Errorf("Missing expected error")
	} else {
		t.Logf("Saw expected panic '%v'", err)
	}

}

// testGotoA1 tests loop-nest-internal goto, no bad iterators.
func testGotoA1(t *testing.T) (result []int, err any) {
	defer func() {
		err = recover()
	}()
	for _, x := range OfSliceIndex([]int{-1, -2, -3, -4, -5}) {
		result = append(result, x)
		if x == -4 {
			break
		}
		defer func() {
			result = append(result, x*10)
		}()
		for _, y := range OfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
			if y == 3 {
				goto A
			}
			result = append(result, y)
		}
		result = append(result, x)
	A:
	}
	return
}

// testGotoA2 tests loop-nest-internal goto, outer bad iterator.
func testGotoA2(t *testing.T) (result []int, err any) {
	defer func() {
		err = recover()
	}()
	for _, x := range BadOfSliceIndex([]int{-1, -2, -3, -4, -5}) {
		result = append(result, x)
		if x == -4 {
			break
		}
		defer func() {
			result = append(result, x*10)
		}()
		for _, y := range OfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
			if y == 3 {
				goto A
			}
			result = append(result, y)
		}
		result = append(result, x)
	A:
	}
	return
}

// testGotoA3 tests loop-nest-internal goto, inner bad iterator.
func testGotoA3(t *testing.T) (result []int, err any) {
	defer func() {
		err = recover()
	}()
	for _, x := range OfSliceIndex([]int{-1, -2, -3, -4, -5}) {
		result = append(result, x)
		if x == -4 {
			break
		}
		defer func() {
			result = append(result, x*10)
		}()
		for _, y := range BadOfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
			if y == 3 {
				goto A
			}
			result = append(result, y)
		}
		result = append(result, x)
	A:
	}
	return
}

func TestGotoA(t *testing.T) {
	var result []int
	var expect = []int{-1, 1, 2, -2, 1, 2, -3, 1, 2, -4, -30, -20, -10}
	var expect3 = []int{-1, 1, 2, -10} // first goto becomes a panic
	var err any

	result, err = testGotoA1(t)
	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}

	result, err = testGotoA2(t)
	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
	if err == nil {
		t.Errorf("Missing expected error")
	} else {
		t.Logf("Saw expected panic '%v'", err)
	}

	result, err = testGotoA3(t)
	if !slices.Equal(expect3, result) {
		t.Errorf("Expected %v, got %v", expect3, result)
	}
	if err == nil {
		t.Errorf("Missing expected error")
	} else {
		t.Logf("Saw expected panic '%v'", err)
	}
}

// testGotoB1 tests loop-nest-exiting goto, no bad iterators.
func testGotoB1(t *testing.T) (result []int, err any) {
	defer func() {
		err = recover()
	}()
	for _, x := range OfSliceIndex([]int{-1, -2, -3, -4, -5}) {
		result = append(result, x)
		if x == -4 {
			break
		}
		defer func() {
			result = append(result, x*10)
		}()
		for _, y := range OfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
			if y == 3 {
				goto B
			}
			result = append(result, y)
		}
		result = append(result, x)
	}
B:
	result = append(result, 999)
	return
}

// testGotoB2 tests loop-nest-exiting goto, outer bad iterator.
func testGotoB2(t *testing.T) (result []int, err any) {
	defer func() {
		err = recover()
	}()
	for _, x := range BadOfSliceIndex([]int{-1, -2, -3, -4, -5}) {
		result = append(result, x)
		if x == -4 {
			break
		}
		defer func() {
			result = append(result, x*10)
		}()
		for _, y := range OfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
			if y == 3 {
				goto B
			}
			result = append(result, y)
		}
		result = append(result, x)
	}
B:
	result = append(result, 999)
	return
}

// testGotoB3 tests loop-nest-exiting goto, inner bad iterator.
func testGotoB3(t *testing.T) (result []int, err any) {
	defer func() {
		err = recover()
	}()
	for _, x := range OfSliceIndex([]int{-1, -2, -3, -4, -5}) {
		result = append(result, x)
		if x == -4 {
			break
		}
		defer func() {
			result = append(result, x*10)
		}()
		for _, y := range BadOfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
			if y == 3 {
				goto B
			}
			result = append(result, y)
		}
		result = append(result, x)
	}
B:
	result = append(result, 999)
	return
}

func TestGotoB(t *testing.T) {
	var result []int
	var expect = []int{-1, 1, 2, 999, -10}
	var expectX = []int{-1, 1, 2, -10}
	var err any

	result, err = testGotoB1(t)
	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}

	result, err = testGotoB2(t)
	if !slices.Equal(expectX, result) {
		t.Errorf("Expected %v, got %v", expectX, result)
	}
	if err == nil {
		t.Errorf("Missing expected error")
	} else {
		t.Logf("Saw expected panic '%v'", err)
	}

	result, err = testGotoB3(t)
	if !slices.Equal(expectX, result) {
		t.Errorf("Expected %v, got %v", expectX, result)
	}
	if err == nil {
		t.Errorf("Missing expected error")
	} else {
		t.Logf("Saw expected panic '%v'", err)
	}
}
