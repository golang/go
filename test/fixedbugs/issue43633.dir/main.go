// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./a"

var g = a.G()

func main() {
	if !a.F() {
		panic("FAIL")
	}
	if !g() {
		panic("FAIL")
	}
}
