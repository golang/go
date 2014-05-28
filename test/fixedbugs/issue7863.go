// run

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

type Foo int64

func (f *Foo) F() int64 {
	return int64(*f)
}

type Bar int64

func (b Bar) F() int64 {
	return int64(b)
}

type Baz int32

func (b Baz) F() int64 {
	return int64(b)
}

func main() {
	foo := Foo(123)
	f := foo.F
	if foo.F() != f() {
		bug()
		fmt.Println("foo.F", foo.F(), f())
	}
	bar := Bar(123)
	f = bar.F
	if bar.F() != f() {
		bug()
		fmt.Println("bar.F", bar.F(), f()) // duh!
	}

	baz := Baz(123)
	f = baz.F
	if baz.F() != f() {
		bug()
		fmt.Println("baz.F", baz.F(), f())
	}
}

var bugged bool

func bug() {
	if !bugged {
		bugged = true
		fmt.Println("BUG")
	}
}
