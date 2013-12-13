// errorcheck

// Used to emit a spurious "invalid recursive type" error.
// See golang.org/issue/5581.

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func NewBar() *Bar { return nil }

func (x *Foo) Method() (int, error) {
	for y := range x.m {
		_ = y.A
	}
	return 0, nil
}

type Foo struct {
	m map[*Bar]int
}

type Bar struct {
	A *Foo
	B chan Blah // ERROR "undefined.*Blah"
}

func main() {
	fmt.Println("Hello, playground")
}
