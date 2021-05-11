// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "sync"

// A _Lockable is a value that may be safely simultaneously accessed
// from multiple goroutines via the Get and Set methods.
type _Lockable[T any] struct {
	T
	mu sync.Mutex
}

// Get returns the value stored in a _Lockable.
func (l *_Lockable[T]) get() T {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.T
}

// set sets the value in a _Lockable.
func (l *_Lockable[T]) set(v T) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.T = v
}

func main() {
	sl := _Lockable[string]{T: "a"}
	if got := sl.get(); got != "a" {
		panic(got)
	}
	sl.set("b")
	if got := sl.get(); got != "b" {
		panic(got)
	}

	il := _Lockable[int]{T: 1}
	if got := il.get(); got != 1 {
		panic(got)
	}
	il.set(2)
	if got := il.get(); got != 2 {
		panic(got)
	}
}
