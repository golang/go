// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "sync"

// A Lockable is a value that may be safely simultaneously accessed
// from multiple goroutines via the Get and Set methods.
type Lockable[T any] struct {
	x  T
	mu sync.Mutex
}

// Get returns the value stored in a Lockable.
func (l *Lockable[T]) get() T {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.x
}

// set sets the value in a Lockable.
func (l *Lockable[T]) set(v T) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.x = v
}

func main() {
	sl := Lockable[string]{x: "a"}
	if got := sl.get(); got != "a" {
		panic(got)
	}
	sl.set("b")
	if got := sl.get(); got != "b" {
		panic(got)
	}

	il := Lockable[int]{x: 1}
	if got := il.get(); got != 1 {
		panic(got)
	}
	il.set(2)
	if got := il.get(); got != 2 {
		panic(got)
	}
}
