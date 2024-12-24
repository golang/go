// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.rangefunc

package race_test

import (
	"runtime"
	"sync/atomic"
	"testing"
)

type Seq2[T1, T2 any] func(yield func(T1, T2) bool)

// ofSliceIndex returns a Seq over the elements of s. It is equivalent
// to range s, except that it splits s into two halves and iterates
// in two separate goroutines.  This is racy if yield is racy, and yield
// will be racy if it contains an early exit.
func ofSliceIndex[T any, S ~[]T](s S) Seq2[int, T] {
	return func(yield func(int, T) bool) {
		c := make(chan bool, 2)
		var done atomic.Bool
		go func() {
			for i := 0; i < len(s)/2; i++ {
				if !done.Load() && !yield(i, s[i]) {
					done.Store(true)
					c <- false
				}
			}
			c <- true
		}()
		go func() {
			for i := len(s) / 2; i < len(s); i++ {
				if !done.Load() && !yield(i, s[i]) {
					done.Store(true)
					c <- false
				}
			}
			c <- true
			return
		}()
		if !<-c {
			return
		}
		<-c
	}
}

// foo is racy, or not, depending on the value of v
// (0-4 == racy, otherwise, not racy).
func foo(v int) int64 {
	var asum atomic.Int64
	for i, x := range ofSliceIndex([]int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
		if i%5 == v {
			break
		}
		asum.Add(x) // don't race on asum
		runtime.Gosched()
	}
	return 100 + asum.Load()
}

// TestRaceRangeFuncIterator races because x%5 can be equal to 4,
// therefore foo can early exit.
func TestRaceRangeFuncIterator(t *testing.T) {
	x := foo(4)
	t.Logf("foo(4)=%d", x)
}

// TestNoRaceRangeFuncIterator does not race because x%5 is never 5,
// therefore foo's loop will not exit early, and this it will not race.
func TestNoRaceRangeFuncIterator(t *testing.T) {
	x := foo(5)
	t.Logf("foo(5)=%d", x)
}
