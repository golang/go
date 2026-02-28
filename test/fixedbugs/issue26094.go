// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "strings"

var X interface{}

type T struct{}

func scopes() {
	p, ok := recover().(error)
	if ok && strings.Contains(p.Error(), "different scopes") {
		return
	}
	panic(p)
}

func F1() {
	type T struct{}
	X = T{}
}

func F2() {
	type T struct{}
	defer scopes()
	_ = X.(T)
}

func F3() {
	defer scopes()
	_ = X.(T)
}

func F4() {
	X = T{}
}

func main() {
	F1() // set X to F1's T
	F2() // check that X is not F2's T
	F3() // check that X is not package T
	F4() // set X to package T
	F2() // check that X is not F2's T
}
