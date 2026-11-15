// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type U interface {
	int | string
	String() string
}

func main() {
	var u U
	println(u == nil)
}
