// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var _ = func() int {
	a = false
	return 0
}()

var a = true
var b = a

func main() {
	if b {
		panic("FAIL")
	}
}
