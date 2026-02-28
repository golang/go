// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure we don't start marshaling (writing to the stack)
// arguments until those arguments are evaluated and known
// not to unconditionally panic. If they unconditionally panic,
// we write some args but never do the call. That messes up
// the logic which decides how big the argout section needs to be.

package main

type W interface {
	Write([]byte)
}

type F func(W)

func foo(f F) {
	defer func() {
		if r := recover(); r != nil {
			usestack(1000)
		}
	}()
	f(nil)
}

func main() {
	foo(func(w W) {
		var x []string
		w.Write([]byte(x[5]))
	})
}

func usestack(n int) {
	if n == 0 {
		return
	}
	usestack(n - 1)
}
