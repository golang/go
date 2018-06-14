// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test interface methods with different return types are distinct.

package main

type S struct { a int }
type T struct { b string }

func (s *S) Name() int8 { return 1 }
func (t *T) Name() int64 { return 64 }

type I1 interface { Name() int8 }
type I2 interface { Name() int64 }

func main() {
	shouldPanic(p1)
}

func p1() {
	var i1 I1
	var s *S
	i1 = s
	print(i1.(I2).Name())
}

func shouldPanic(f func()) {
	defer func() {
		if recover() == nil {
			panic("function should panic")
		}
	}()
	f()
}
