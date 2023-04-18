// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var c = b
var d = a

var a, b any = any(nil).(bool)

func main() {
	if c != false {
		panic(c)
	}
	if d != false {
		panic(d)
	}
}
