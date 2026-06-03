// compile

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var a, b, c interface{} = func() (_, _, _ int) { return 1, 2, 3 }()

func main() {
	println(a.(int), b.(int), c.(int))
}
