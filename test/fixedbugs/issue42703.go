// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var ok [2]bool

func main() {
	f()()
	if !ok[0] || !ok[1] {
		panic("FAIL")
	}
}

func f() func() { ok[0] = true; return g }
func g()        { ok[1] = true }
