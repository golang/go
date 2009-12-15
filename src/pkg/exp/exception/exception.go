// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package illustrates how basic try-catch exception handling
// can be emulated using goroutines, channels, and closures.
//
// This package is *not* intended as a general exception handler
// library.
//
package exception

import (
	"fmt"
	"runtime"
)

// A Handler function handles an arbitrary exception value x.
type Handler func(x interface{})

// An Exception carries an exception value.
type Exception struct {
	Value interface{} // Value may be the nil exception
}

// Try invokes a function f with a Handler to throw exceptions.
// The function f may terminate abnormally with an arbitrary
// exception x by calling throw(x) within f. If an exception is
// thrown, Try returns an *Exception; otherwise it returns nil.
//
// Usage pattern:
//
//	if x := exception.Try(func(throw exception.Handler) {
//		...
//		throw(42);  // terminate f by throwing exception 42
//		...
//	}); x != nil {
//		// catch exception, e.g. print it
//		fmt.Println(x.Value);
//	}
//
// Alternative:
//
//	exception.Try(func(throw exception.Handler) {
//		...
//		throw(42);  // terminate f by throwing exception 42
//		...
//	}).Catch(func (x interface{}) {
//		// catch exception, e.g. print it
//		fmt.Println(x);
//	})
//
func Try(f func(throw Handler)) *Exception {
	h := make(chan *Exception)

	// execute try block
	go func() {
		f(func(x interface{}) {
			h <- &Exception{x}
			runtime.Goexit()
		})
		h <- nil // clean termination
	}()

	return <-h
}


// If x != nil, Catch invokes f with the exception value x.Value.
// See Try for usage patterns.
func (x *Exception) Catch(f Handler) {
	if x != nil {
		f(x.Value)
	}
}


func (x *Exception) String() string {
	if x != nil {
		return fmt.Sprintf("exception: %v", x.Value)
	}
	return ""
}
