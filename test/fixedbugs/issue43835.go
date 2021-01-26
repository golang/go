// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	if f() {
		panic("FAIL")
	}
	if bad, _ := g(); bad {
		panic("FAIL")
	}
	if bad, _ := h(); bad {
		panic("FAIL")
	}
}

func f() (bad bool) {
	defer func() {
		recover()
	}()
	var p *int
	bad, _ = true, *p
	return
}

func g() (bool, int) {
	defer func() {
		recover()
	}()
	var p *int
	return true, *p
}


func h() (_ bool, _ int) {
	defer func() {
		recover()
	}()
	var p *int
	return true, *p
}
