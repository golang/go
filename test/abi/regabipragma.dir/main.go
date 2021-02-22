// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"regabipragma.dir/tmp"
)

type S string

//go:noinline
func (s S) ff(t string) string {
	return string(s) + " " + t
}

//go:noinline
//go:registerparams
func f(s,t string) string { // ERROR "Declared function f has register params"
	return s + " " + t
}

func check(s string) {
	if s != "Hello world!" {
		fmt.Printf("FAIL, wanted 'Hello world!' but got '%s'\n", s)
	}
}

func main() {
	check(f("Hello", "world!"))   // ERROR "Called function ...f has register params"
	check(tmp.F("Hello", "world!"))  // ERROR "Called function regabipragma.dir/tmp.F has register params"
	check(S("Hello").ff("world!"))
	check(tmp.S("Hello").FF("world!"))
}
