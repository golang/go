// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./a"

func main() {
	var x a.I = &a.T{}
	x.M() // call to the wrapper (*T).M
	a.F() // make sure a.F is not dead, which also calls (*T).M inside package a
}
