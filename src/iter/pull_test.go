// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iter_test

import (
	"fmt"
	. "iter"
	"runtime"
	"testing"
)

func count(n int) Seq[int] {
	return func(yield func(int) bool) {
		for i := range n {
			if !yield(i) {
				break
			}
		}
	}
}

func squares(n int) Seq2[int, int64] {
	return func(yield func(int, int64) bool) {
		for i := range n {
			if !yield(i, int64(i)*int64(i)) {
				break
			}
		}
	}
}

func TestPull(t *testing.T) {
	for end := 0; end <= 3; end++ {
		t.Run(fmt.Sprint(end), func(t *testing.T) {
			ng := runtime.NumGoroutine()
			wantNG := func(want int) {
				if xg := runtime.NumGoroutine() - ng; xg != want {
					t.Helper()
					t.Errorf("have %d extra goroutines, want %d", xg, want)
				}
			}
			wantNG(0)
			next, stop := Pull(count(3))
			wantNG(1)
			for i := range end {
				v, ok := next()
				if v != i || ok != true {
					t.Fatalf("next() = %d, %v, want %d, %v", v, ok, i, true)
				}
				wantNG(1)
			}
			wantNG(1)
			if end < 3 {
				stop()
				wantNG(0)
			}
			for range 2 {
				v, ok := next()
				if v != 0 || ok != false {
					t.Fatalf("next() = %d, %v, want %d, %v", v, ok, 0, false)
				}
				wantNG(0)
			}
			wantNG(0)

			stop()
			stop()
			stop()
			wantNG(0)
		})
	}
}

func TestPull2(t *testing.T) {
	for end := 0; end <= 3; end++ {
		t.Run(fmt.Sprint(end), func(t *testing.T) {
			ng := runtime.NumGoroutine()
			wantNG := func(want int) {
				if xg := runtime.NumGoroutine() - ng; xg != want {
					t.Helper()
					t.Errorf("have %d extra goroutines, want %d", xg, want)
				}
			}
			wantNG(0)
			next, stop := Pull2(squares(3))
			wantNG(1)
			for i := range end {
				k, v, ok := next()
				if k != i || v != int64(i*i) || ok != true {
					t.Fatalf("next() = %d, %d, %v, want %d, %d, %v", k, v, ok, i, i*i, true)
				}
				wantNG(1)
			}
			wantNG(1)
			if end < 3 {
				stop()
				wantNG(0)
			}
			for range 2 {
				k, v, ok := next()
				if v != 0 || ok != false {
					t.Fatalf("next() = %d, %d, %v, want %d, %d, %v", k, v, ok, 0, 0, false)
				}
				wantNG(0)
			}
			wantNG(0)

			stop()
			stop()
			stop()
			wantNG(0)
		})
	}
}

func TestPullDoubleNext(t *testing.T) {
	next, _ := Pull(doDoubleNext())
	nextSlot = next
	next()
	if nextSlot != nil {
		t.Fatal("double next did not fail")
	}
}

var nextSlot func() (int, bool)

func doDoubleNext() Seq[int] {
	return func(_ func(int) bool) {
		defer func() {
			if recover() != nil {
				nextSlot = nil
			}
		}()
		nextSlot()
	}
}

func TestPullDoubleNext2(t *testing.T) {
	next, _ := Pull2(doDoubleNext2())
	nextSlot2 = next
	next()
	if nextSlot2 != nil {
		t.Fatal("double next did not fail")
	}
}

var nextSlot2 func() (int, int, bool)

func doDoubleNext2() Seq2[int, int] {
	return func(_ func(int, int) bool) {
		defer func() {
			if recover() != nil {
				nextSlot2 = nil
			}
		}()
		nextSlot2()
	}
}

func TestPullDoubleYield(t *testing.T) {
	_, stop := Pull(storeYield())
	defer func() {
		if recover() != nil {
			yieldSlot = nil
		}
		stop()
	}()
	yieldSlot(5)
	if yieldSlot != nil {
		t.Fatal("double yield did not fail")
	}
}

func storeYield() Seq[int] {
	return func(yield func(int) bool) {
		yieldSlot = yield
		if !yield(5) {
			return
		}
	}
}

var yieldSlot func(int) bool

func TestPullDoubleYield2(t *testing.T) {
	_, stop := Pull2(storeYield2())
	defer func() {
		if recover() != nil {
			yieldSlot2 = nil
		}
		stop()
	}()
	yieldSlot2(23, 77)
	if yieldSlot2 != nil {
		t.Fatal("double yield did not fail")
	}
}

func storeYield2() Seq2[int, int] {
	return func(yield func(int, int) bool) {
		yieldSlot2 = yield
		if !yield(23, 77) {
			return
		}
	}
}

var yieldSlot2 func(int, int) bool
