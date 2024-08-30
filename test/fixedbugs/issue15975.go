// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var fail bool

type Closer interface {
	Close()
}

func nilInterfaceDeferCall() {
	var x Closer
	defer x.Close()
	// if it panics when evaluating x.Close, it should not reach here
	fail = true
}

func shouldPanic(f func()) {
	defer func() {
		if recover() == nil {
			panic("did not panic")
		}
	}()
	f()
}

func main() {
	shouldPanic(nilInterfaceDeferCall)
	if fail {
		panic("fail")
	}
}
