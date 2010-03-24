// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"fmt"
	"os"
	"runtime"
)

// Abort aborts the thread's current computation,
// causing the innermost Try to return err.
func (t *Thread) Abort(err os.Error) {
	if t.abort == nil {
		panic("abort: " + err.String())
	}
	t.abort <- err
	runtime.Goexit()
}

// Try executes a computation; if the computation
// Aborts, Try returns the error passed to abort.
func (t *Thread) Try(f func(t *Thread)) os.Error {
	oc := t.abort
	c := make(chan os.Error)
	t.abort = c
	go func() {
		f(t)
		c <- nil
	}()
	err := <-c
	t.abort = oc
	return err
}

type DivByZeroError struct{}

func (DivByZeroError) String() string { return "divide by zero" }

type NilPointerError struct{}

func (NilPointerError) String() string { return "nil pointer dereference" }

type IndexError struct {
	Idx, Len int64
}

func (e IndexError) String() string {
	if e.Idx < 0 {
		return fmt.Sprintf("negative index: %d", e.Idx)
	}
	return fmt.Sprintf("index %d exceeds length %d", e.Idx, e.Len)
}

type SliceError struct {
	Lo, Hi, Cap int64
}

func (e SliceError) String() string {
	return fmt.Sprintf("slice [%d:%d]; cap %d", e.Lo, e.Hi, e.Cap)
}

type KeyError struct {
	Key interface{}
}

func (e KeyError) String() string { return fmt.Sprintf("key '%v' not found in map", e.Key) }

type NegativeLengthError struct {
	Len int64
}

func (e NegativeLengthError) String() string {
	return fmt.Sprintf("negative length: %d", e.Len)
}

type NegativeCapacityError struct {
	Len int64
}

func (e NegativeCapacityError) String() string {
	return fmt.Sprintf("negative capacity: %d", e.Len)
}
