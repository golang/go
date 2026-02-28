// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test for declaration and use of a parameterized embedded field.

package main

import (
	"fmt"
	"sync"
)

type MyStruct[T any] struct {
	val T
}

type Lockable[T any] struct {
	MyStruct[T]
	mu sync.Mutex
}

// Get returns the value stored in a Lockable.
func (l *Lockable[T]) Get() T {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.MyStruct.val
}

// Set sets the value in a Lockable.
func (l *Lockable[T]) Set(v T) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.MyStruct = MyStruct[T]{v}
}

func main() {
	var li Lockable[int]

	li.Set(5)
	if got, want := li.Get(), 5; got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}
}
