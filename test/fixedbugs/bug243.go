// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "os"

// Issue 481: closures and var declarations
// with multiple variables assigned from one
// function call.

func main() {
	var listen, _ = Listen("tcp", "127.0.0.1:0")

	go func() {
		for {
			var conn, _ = listen.Accept()
			_ = conn
		}
	}()

	var conn, _ = Dial("tcp", "", listen.Addr().String())
	_ = conn
}

// Simulated net interface to exercise bug
// without involving a real network.
type T chan int

var global T

func Listen(x, y string) (T, string) {
	global = make(chan int)
	return global, y
}

func (t T) Addr() os.Error {
	return os.NewError("stringer")
}

func (t T) Accept() (int, string) {
	return <-t, ""
}

func Dial(x, y, z string) (int, string) {
	global <- 1
	return 0, ""
}
