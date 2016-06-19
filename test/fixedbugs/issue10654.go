// compile

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 10654: Failure to use generated temps
// for function calls etc. in boolean codegen.

package main

var s string

func main() {
	if (s == "this") != (s == "that") {
	}
}
