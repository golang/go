// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rangefunc_test

import (
	"fmt"
	"regexp"
	"slices"
	"testing"
)

type Seq[T any] func(yield func(T) bool)
type Seq2[T1, T2 any] func(yield func(T1, T2) bool)

// OfSliceIndex returns a Seq2 over the elements of s. It is equivalent
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

// SwallowPanicOfSliceIndex hides panics and converts them to normal return
func SwallowPanicOfSliceIndex[T any, S ~[]T](s S) Seq2[int, T] {
	return func(yield func(int, T) bool) {
		for i, v := range s {
			done := false
			func() {
				defer func() {
					if r := recover(); r != nil {
						done = true
					}
				}()
				done = !yield(i, v)
			}()
			if done {
				return
			}
		}
		return
	}
}

// PanickyOfSliceIndex iterates the slice but panics if it exits the loop early
func PanickyOfSliceIndex[T any, S ~[]T](s S) Seq2[int, T] {
	return func(yield func(int, T) bool) {
		for i, v := range s {
			if !yield(i, v) {
				panic(fmt.Errorf("Panicky iterator panicking"))
			}
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

func (ti *TrickyIterator) iterEcho(s []int) Seq2[int, int] {
	return func(yield func(int, int) bool) {
		for i, v := range s {
			if !yield(i, v) {
				ti.yield = yield
				return
			}
			if ti.yield != nil && !ti.yield(i, v) {
				return
			}
		}
		ti.yield = yield
		return
	}
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

const DONE = 0      // body of loop has exited in a non-panic way
const READY = 1     // body of loop has not exited yet, is not running
const PANIC = 2     // body of loop is either currently running, or has panicked
const EXHAUSTED = 3 // iterator function return, i.e., sequence is "exhausted"

const MISSING_PANIC = 4 // overload "READY" for panic call

// Check2 wraps the function body passed to iterator forall
// in code that ensures that it cannot (successfully) be called
// either after body return false (control flow out of loop) or
// forall itself returns (the iteration is now done).
//
// Note that this can catch errors before the inserted checks.
func Check2[U, V any](forall Seq2[U, V]) Seq2[U, V] {
	return func(body func(U, V) bool) {
		state := READY
		forall(func(u U, v V) bool {
			if state != READY {
				panic(fail[state])
			}
			state = PANIC
			ret := body(u, v)
			if ret {
				state = READY
			} else {
				state = DONE
			}
			return ret
		})
		if state == PANIC {
			panic(fail[MISSING_PANIC])
		}
		state = EXHAUSTED
	}
}

func Check[U any](forall Seq[U]) Seq[U] {
	return func(body func(U) bool) {
		state := READY
		forall(func(u U) bool {
			if state != READY {
				panic(fail[state])
			}
			state = PANIC
			ret := body(u)
			if ret {
				state = READY
			} else {
				state = DONE
			}
			return ret
		})
		if state == PANIC {
			panic(fail[MISSING_PANIC])
		}
		state = EXHAUSTED
	}
}

func matchError(r any, x string) bool {
	if r == nil {
		return false
	}
	if x == "" {
		return true
	}
	if p, ok := r.(errorString); ok {
		return p.Error() == x
	}
	if p, ok := r.(error); ok {
		e, err := regexp.Compile(x)
		if err != nil {
			panic(fmt.Errorf("Bad regexp '%s' passed to matchError", x))
		}
		return e.MatchString(p.Error())
	}
	return false
}

func matchErrorHelper(t *testing.T, r any, x string) {
	if matchError(r, x) {
		t.Logf("Saw expected panic '%v'", r)
	} else {
		t.Errorf("Saw wrong panic '%v', expected '%s'", r, x)
	}
}

// An errorString represents a runtime error described by a single string.
type errorString string

func (e errorString) Error() string {
	return string(e)
}

const (
	// RERR_ is for runtime error, and may be regexps/substrings, to simplify use of tests with tools
	RERR_DONE      = "runtime error: range function continued iteration after function for loop body returned false"
	RERR_PANIC     = "runtime error: range function continued iteration after loop body panic"
	RERR_EXHAUSTED = "runtime error: range function continued iteration after whole loop exit"
	RERR_MISSING   = "runtime error: range function recovered a loop body panic and did not resume panicking"

	// CERR_ is for checked errors in the Check combinator defined above, and should be literal strings
	CERR_PFX       = "checked rangefunc error: "
	CERR_DONE      = CERR_PFX + "loop iteration after body done"
	CERR_PANIC     = CERR_PFX + "loop iteration after panic"
	CERR_EXHAUSTED = CERR_PFX + "loop iteration after iterator exit"
	CERR_MISSING   = CERR_PFX + "loop iterator swallowed panic"
)

var fail []error = []error{
	errorString(CERR_DONE),
	errorString(CERR_PFX + "loop iterator, unexpected error"),
	errorString(CERR_PANIC),
	errorString(CERR_EXHAUSTED),
	errorString(CERR_MISSING),
}

func TestCheck(t *testing.T) {
	i := 0
	defer func() {
		if r := recover(); r != nil {
			if matchError(r, CERR_DONE) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
		} else {
			t.Error("Wanted to see a failure")
		}
	}()
	for _, x := range Check2(BadOfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})) {
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
		if matchError(r, RERR_EXHAUSTED) {
			t.Logf("Saw expected panic '%v'", r)
		} else {
			t.Errorf("Saw wrong panic '%v'", r)
		}
	} else {
		t.Error("Wanted to see a failure")
	}
	if i != 36 {
		t.Errorf("Expected i == 36, saw %d instead", i)
	} else {
		t.Logf("i = %d", i)
	}
}

func TestCooperativeBadOfSliceIndexCheck(t *testing.T) {
	i := 0
	proceed := make(chan any)
	for _, x := range Check2(CooperativeBadOfSliceIndex([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, proceed)) {
		i += x
		if i >= 36 {
			break
		}
	}
	proceed <- true
	if r := <-proceed; r != nil {
		if matchError(r, CERR_EXHAUSTED) {
			t.Logf("Saw expected panic '%v'", r)
		} else {
			t.Errorf("Saw wrong panic '%v'", r)
		}

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
			if matchError(r, RERR_EXHAUSTED) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
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
			if matchError(r, RERR_EXHAUSTED) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
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
			if matchError(r, RERR_EXHAUSTED) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
		} else {
			t.Error("Wanted to see a failure")
		}
	}()

	trickItZero.fail()
}

func TestTrickyIterZeroCheck(t *testing.T) {
	trickItZero := TrickyIterator{}
	i := 0
	for _, x := range Check2(trickItZero.iterZero([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})) {
		i += x
		if i >= 36 {
			break
		}
	}

	// Don't care about value, ought to be 0 anyhow.
	t.Logf("i = %d", i)

	defer func() {
		if r := recover(); r != nil {
			if matchError(r, CERR_EXHAUSTED) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
		} else {
			t.Error("Wanted to see a failure")
		}
	}()

	trickItZero.fail()
}

func TestTrickyIterEcho(t *testing.T) {
	trickItAll := TrickyIterator{}
	i := 0
	for _, x := range trickItAll.iterAll([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
		t.Logf("first loop i=%d", i)
		i += x
		if i >= 10 {
			break
		}
	}

	if i != 10 {
		t.Errorf("Expected i == 10, saw %d instead", i)
	} else {
		t.Logf("i = %d", i)
	}

	defer func() {
		if r := recover(); r != nil {
			if matchError(r, RERR_EXHAUSTED) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
		} else {
			t.Error("Wanted to see a failure")
		}
	}()

	i = 0
	for _, x := range trickItAll.iterEcho([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
		t.Logf("second loop i=%d", i)
		if x >= 5 {
			break
		}
	}

}

func TestTrickyIterEcho2(t *testing.T) {
	trickItAll := TrickyIterator{}
	var i int

	defer func() {
		if r := recover(); r != nil {
			if matchError(r, RERR_EXHAUSTED) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
		} else {
			t.Error("Wanted to see a failure")
		}
	}()

	for k := range 2 {
		i = 0
		for _, x := range trickItAll.iterEcho([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
			t.Logf("k,x,i=%d,%d,%d", k, x, i)
			i += x
			if i >= 10 {
				break
			}
		}
		t.Logf("i = %d", i)

		if i != 10 {
			t.Errorf("Expected i == 10, saw %d instead", i)
		}
	}
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
			if matchError(r, RERR_DONE) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
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
			if matchError(r, RERR_DONE) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
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
			if matchError(r, RERR_DONE) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
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
			if matchError(r, RERR_DONE) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
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
			if matchError(r, RERR_DONE) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
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
			if matchError(r, RERR_DONE) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
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
			if matchError(r, RERR_DONE) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
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
			if matchError(r, RERR_DONE) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
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
			if matchError(r, RERR_DONE) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
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

func TestPanickyIterator1(t *testing.T) {
	var result []int
	var expect = []int{1, 2, 3, 4}
	defer func() {
		if r := recover(); r != nil {
			if matchError(r, "Panicky iterator panicking") {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
		} else {
			t.Errorf("Wanted to see a failure, result was %v", result)
		}
		if !slices.Equal(expect, result) {
			t.Errorf("Expected %v, got %v", expect, result)
		}
	}()
	for _, z := range PanickyOfSliceIndex([]int{1, 2, 3, 4}) {
		result = append(result, z)
		if z == 4 {
			break
		}
	}
}

func TestPanickyIterator1Check(t *testing.T) {
	var result []int
	var expect = []int{1, 2, 3, 4}
	defer func() {
		if r := recover(); r != nil {
			if matchError(r, "Panicky iterator panicking") {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
			if !slices.Equal(expect, result) {
				t.Errorf("Expected %v, got %v", expect, result)
			}
		} else {
			t.Errorf("Wanted to see a failure, result was %v", result)
		}
	}()
	for _, z := range Check2(PanickyOfSliceIndex([]int{1, 2, 3, 4})) {
		result = append(result, z)
		if z == 4 {
			break
		}
	}
}

func TestPanickyIterator2(t *testing.T) {
	var result []int
	var expect = []int{100, 10, 1, 2}
	defer func() {
		if r := recover(); r != nil {
			if matchError(r, RERR_MISSING) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
		} else {
			t.Errorf("Wanted to see a failure, result was %v", result)
		}
		if !slices.Equal(expect, result) {
			t.Errorf("Expected %v, got %v", expect, result)
		}
	}()
	for _, x := range OfSliceIndex([]int{100, 200}) {
		result = append(result, x)
	Y:
		// swallows panics and iterates to end BUT `break Y` disables the body, so--> 10, 1, 2
		for _, y := range VeryBadOfSliceIndex([]int{10, 20}) {
			result = append(result, y)

			// converts early exit into a panic --> 1, 2
			for k, z := range PanickyOfSliceIndex([]int{1, 2}) { // iterator panics
				result = append(result, z)
				if k == 1 {
					break Y
				}
			}
		}
	}
}

func TestPanickyIterator2Check(t *testing.T) {
	var result []int
	var expect = []int{100, 10, 1, 2}
	defer func() {
		if r := recover(); r != nil {
			if matchError(r, CERR_MISSING) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
		} else {
			t.Errorf("Wanted to see a failure, result was %v", result)
		}
		if !slices.Equal(expect, result) {
			t.Errorf("Expected %v, got %v", expect, result)
		}
	}()
	for _, x := range Check2(OfSliceIndex([]int{100, 200})) {
		result = append(result, x)
	Y:
		// swallows panics and iterates to end BUT `break Y` disables the body, so--> 10, 1, 2
		for _, y := range Check2(VeryBadOfSliceIndex([]int{10, 20})) {
			result = append(result, y)

			// converts early exit into a panic --> 1, 2
			for k, z := range Check2(PanickyOfSliceIndex([]int{1, 2})) { // iterator panics
				result = append(result, z)
				if k == 1 {
					break Y
				}
			}
		}
	}
}

func TestPanickyIterator3(t *testing.T) {
	var result []int
	var expect = []int{100, 10, 1, 2, 200, 10, 1, 2}
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Unexpected panic '%v'", r)
		}
		if !slices.Equal(expect, result) {
			t.Errorf("Expected %v, got %v", expect, result)
		}
	}()
	for _, x := range OfSliceIndex([]int{100, 200}) {
		result = append(result, x)
	Y:
		// swallows panics and iterates to end BUT `break Y` disables the body, so--> 10, 1, 2
		// This is cross-checked against the checked iterator below; the combinator should behave the same.
		for _, y := range VeryBadOfSliceIndex([]int{10, 20}) {
			result = append(result, y)

			for k, z := range OfSliceIndex([]int{1, 2}) { // iterator does not panic
				result = append(result, z)
				if k == 1 {
					break Y
				}
			}
		}
	}
}
func TestPanickyIterator3Check(t *testing.T) {
	var result []int
	var expect = []int{100, 10, 1, 2, 200, 10, 1, 2}
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Unexpected panic '%v'", r)
		}
		if !slices.Equal(expect, result) {
			t.Errorf("Expected %v, got %v", expect, result)
		}
	}()
	for _, x := range Check2(OfSliceIndex([]int{100, 200})) {
		result = append(result, x)
	Y:
		// swallows panics and iterates to end BUT `break Y` disables the body, so--> 10, 1, 2
		for _, y := range Check2(VeryBadOfSliceIndex([]int{10, 20})) {
			result = append(result, y)

			for k, z := range Check2(OfSliceIndex([]int{1, 2})) { // iterator does not panic
				result = append(result, z)
				if k == 1 {
					break Y
				}
			}
		}
	}
}

func TestPanickyIterator4(t *testing.T) {
	var result []int
	var expect = []int{1, 2, 3}
	defer func() {
		if r := recover(); r != nil {
			if matchError(r, RERR_MISSING) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
		}
		if !slices.Equal(expect, result) {
			t.Errorf("Expected %v, got %v", expect, result)
		}
	}()
	for _, x := range SwallowPanicOfSliceIndex([]int{1, 2, 3, 4}) {
		result = append(result, x)
		if x == 3 {
			panic("x is 3")
		}
	}

}
func TestPanickyIterator4Check(t *testing.T) {
	var result []int
	var expect = []int{1, 2, 3}
	defer func() {
		if r := recover(); r != nil {
			if matchError(r, CERR_MISSING) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
		}
		if !slices.Equal(expect, result) {
			t.Errorf("Expected %v, got %v", expect, result)
		}
	}()
	for _, x := range Check2(SwallowPanicOfSliceIndex([]int{1, 2, 3, 4})) {
		result = append(result, x)
		if x == 3 {
			panic("x is 3")
		}
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

// veryBadCheck wraps a "very bad" iterator with Check,
// demonstrating that the very bad iterator also hides panics
// thrown by Check.
func veryBadCheck(s []int) []int {
	var result []int
X:
	for _, x := range OfSliceIndex([]int{1, 2, 3}) {

		result = append(result, x)

		for _, y := range Check2(VeryBadOfSliceIndex(s)) {
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

// TestVeryBadCheck checks the behavior of an extremely poorly behaved iterator,
// which also suppresses the exceptions from "Check"
func TestVeryBadCheck(t *testing.T) {
	result := veryBadCheck([]int{10, 20, 30, 40}) // even length
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
			if matchError(r, RERR_DONE) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
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

// testReturn4 has no bad iterators, but exercises  return variable rewriting
// differs from testReturn1 because deferred append to "result" does not change
// the return value in this case.
func testReturn4(t *testing.T) (_ []int, _ []int, err any) {
	var result []int
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
				return result, result, nil
			}
			result = append(result, y)
		}
		result = append(result, x)
	}
	return
}

// TestReturns checks that returns through bad iterators behave properly,
// for inner and outer bad iterators.
func TestReturns(t *testing.T) {
	var result []int
	var result2 []int
	var expect = []int{-1, 1, 2, -10}
	var expect2 = []int{-1, 1, 2}
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
		if matchError(err, RERR_DONE) {
			t.Logf("Saw expected panic '%v'", err)
		} else {
			t.Errorf("Saw wrong panic '%v'", err)
		}
	}

	result, err = testReturn3(t)
	if !slices.Equal(expect, result) {
		t.Errorf("Expected %v, got %v", expect, result)
	}
	if err == nil {
		t.Errorf("Missing expected error")
	} else {
		if matchError(err, RERR_DONE) {
			t.Logf("Saw expected panic '%v'", err)
		} else {
			t.Errorf("Saw wrong panic '%v'", err)
		}
	}

	result, result2, err = testReturn4(t)
	if !slices.Equal(expect2, result) {
		t.Errorf("Expected %v, got %v", expect2, result)
	}
	if !slices.Equal(expect2, result2) {
		t.Errorf("Expected %v, got %v", expect2, result2)
	}
	if err != nil {
		t.Errorf("Unexpected error %v", err)
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
		if matchError(err, RERR_DONE) {
			t.Logf("Saw expected panic '%v'", err)
		} else {
			t.Errorf("Saw wrong panic '%v'", err)
		}
	}

	result, err = testGotoA3(t)
	if !slices.Equal(expect3, result) {
		t.Errorf("Expected %v, got %v", expect3, result)
	}
	if err == nil {
		t.Errorf("Missing expected error")
	} else {
		if matchError(err, RERR_DONE) {
			t.Logf("Saw expected panic '%v'", err)
		} else {
			t.Errorf("Saw wrong panic '%v'", err)
		}
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
		if matchError(err, RERR_DONE) {
			t.Logf("Saw expected panic '%v'", err)
		} else {
			t.Errorf("Saw wrong panic '%v'", err)
		}
	}

	result, err = testGotoB3(t)
	if !slices.Equal(expectX, result) {
		t.Errorf("Expected %v, got %v", expectX, result)
	}
	if err == nil {
		t.Errorf("Missing expected error")
	} else {
		matchErrorHelper(t, err, RERR_DONE)
	}
}

// once returns an iterator that runs its loop body once with the supplied value
func once[T any](x T) Seq[T] {
	return func(yield func(T) bool) {
		yield(x)
	}
}

// terrify converts an iterator into one that panics with the supplied string
// if/when the loop body terminates early (returns false, for break, goto, outer
// continue, or return).
func terrify[T any](s string, forall Seq[T]) Seq[T] {
	return func(yield func(T) bool) {
		forall(func(v T) bool {
			if !yield(v) {
				panic(s)
			}
			return true
		})
	}
}

func use[T any](T) {
}

// f runs a not-rangefunc iterator that recovers from a panic that follows execution of a return.
// what does f return?
func f() string {
	defer func() { recover() }()
	defer panic("f panic")
	for _, s := range []string{"f return"} {
		return s
	}
	return "f not reached"
}

// g runs a rangefunc iterator that recovers from a panic that follows execution of a return.
// what does g return?
func g() string {
	defer func() { recover() }()
	for s := range terrify("g panic", once("g return")) {
		return s
	}
	return "g not reached"
}

// h runs a rangefunc iterator that recovers from a panic that follows execution of a return.
// the panic occurs in the rangefunc iterator itself.
// what does h return?
func h() (hashS string) {
	defer func() { recover() }()
	for s := range terrify("h panic", once("h return")) {
		hashS := s
		use(hashS)
		return s
	}
	return "h not reached"
}

func j() (hashS string) {
	defer func() { recover() }()
	for s := range terrify("j panic", once("j return")) {
		hashS = s
		return
	}
	return "j not reached"
}

// k runs a rangefunc iterator that recovers from a panic that follows execution of a return.
// the panic occurs in the rangefunc iterator itself.
// k includes an additional mechanism to for making the return happen
// what does k return?
func k() (hashS string) {
	_return := func(s string) { hashS = s }

	defer func() { recover() }()
	for s := range terrify("k panic", once("k return")) {
		_return(s)
		return
	}
	return "k not reached"
}

func m() (hashS string) {
	_return := func(s string) { hashS = s }

	defer func() { recover() }()
	for s := range terrify("m panic", once("m return")) {
		defer _return(s)
		return s + ", but should be replaced in a defer"
	}
	return "m not reached"
}

func n() string {
	defer func() { recover() }()
	for s := range terrify("n panic", once("n return")) {
		return s + func(s string) string {
			defer func() { recover() }()
			for s := range terrify("n closure panic", once(s)) {
				return s
			}
			return "n closure not reached"
		}(" and n closure return")
	}
	return "n not reached"
}

type terrifyTestCase struct {
	f func() string
	e string
}

func TestPanicReturns(t *testing.T) {
	tcs := []terrifyTestCase{
		{f, "f return"},
		{g, "g return"},
		{h, "h return"},
		{k, "k return"},
		{j, "j return"},
		{m, "m return"},
		{n, "n return and n closure return"},
	}

	for _, tc := range tcs {
		got := tc.f()
		if got != tc.e {
			t.Errorf("Got %s expected %s", got, tc.e)
		} else {
			t.Logf("Got expected %s", got)
		}
	}
}

// twice calls yield twice, the first time defer-recover-saving any panic,
// for re-panicking later if the second call to yield does not also panic.
// If the first call panicked, the second call ought to also panic because
// it was called after a panic-termination of the loop body.
func twice[T any](x, y T) Seq[T] {
	return func(yield func(T) bool) {
		var p any
		done := false
		func() {
			defer func() {
				p = recover()
			}()
			done = !yield(x)
		}()
		if done {
			return
		}
		yield(y)
		if p != nil {
			// do not swallow the panic
			panic(p)
		}
	}
}

func TestRunBodyAfterPanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			if matchError(r, RERR_PANIC) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
		} else {
			t.Errorf("Wanted to see a failure, result")
		}
	}()
	for x := range twice(0, 1) {
		if x == 0 {
			panic("x is zero")
		}
	}
}

func TestRunBodyAfterPanicCheck(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			if matchError(r, CERR_PANIC) {
				t.Logf("Saw expected panic '%v'", r)
			} else {
				t.Errorf("Saw wrong panic '%v'", r)
			}
		} else {
			t.Errorf("Wanted to see a failure, result")
		}
	}()
	for x := range Check(twice(0, 1)) {
		if x == 0 {
			panic("x is zero")
		}
	}
}

func TestTwoLevelReturn(t *testing.T) {
	f := func() int {
		for a := range twice(0, 1) {
			for b := range twice(0, 2) {
				x := a + b
				t.Logf("x=%d", x)
				if x == 3 {
					return x
				}
			}
		}
		return -1
	}
	y := f()
	if y != 3 {
		t.Errorf("Expected y=3, got y=%d\n", y)
	}
}

func TestTwoLevelReturnCheck(t *testing.T) {
	f := func() int {
		for a := range Check(twice(0, 1)) {
			for b := range Check(twice(0, 2)) {
				x := a + b
				t.Logf("a=%d, b=%d, x=%d", a, b, x)
				if x == 3 {
					return x
				}
			}
		}
		return -1
	}
	y := f()
	if y != 3 {
		t.Errorf("Expected y=3, got y=%d\n", y)
	}
}
