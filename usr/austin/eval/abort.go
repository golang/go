// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"fmt";
	"os";
	"runtime";
)

// TODO(austin) This is not thread-safe.  We could include the abort
// channel in the Frame structure, but then the Value methods need to
// take the Frame.  However, passing something to the Value methods
// might be necessary to generate back traces.
var abortChan = make(chan os.Error)

// Abort aborts the current computation.  If this is called within the
// extent of a Try call, this immediately returns to the Try with the
// given error.  If not, then this panic's.
func Abort(e os.Error) {
	if abortChan == nil {
		panic("Abort: " + e.String());
	}
	abortChan <- e;
	runtime.Goexit();
}

// Try executes a computation with the ability to Abort.
func Try(f func()) os.Error {
	abortChan = make(chan os.Error);
	go func() {
		f();
		abortChan <- nil;
	}();
	res := <-abortChan;
	abortChan = nil;
	return res;
}

type DivByZeroError struct {}

func (DivByZeroError) String() string {
	return "divide by zero";
}

type NilPointerError struct {}

func (NilPointerError) String() string {
	return "nil pointer dereference";
}

type IndexError struct {
	Idx, Len int64;
}

func (e IndexError) String() string {
	if e.Idx < 0 {
		return fmt.Sprintf("negative index: %d", e.Idx);
	}
	return fmt.Sprintf("index %d exceeds length %d", e.Idx, e.Len);
}

type KeyError struct {
	Key interface {};
}

func (e KeyError) String() string {
	return fmt.Sprintf("key '%v' not found in map", e.Key);
}

type NegativeLengthError struct {
	Len int64;
}

func (e NegativeLengthError) String() string {
	return fmt.Sprintf("negative length: %d", e.Len);
}

type NegativeCapacityError struct {
	Len int64;
}

func (e NegativeCapacityError) String() string {
	return fmt.Sprintf("negative capacity: %d", e.Len);
}
