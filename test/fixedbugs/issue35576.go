// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check print/println(f()) is allowed where f() is multi-value.

package main

func f() (int16, float64, string) { return -42, 42.0, "x" }

func main() {
	print(f())
	println(f())
}
