// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f() /* no return type */ {}

func main() {
	x := f();  // ERROR "mismatch|as value|no type"
	_ = x
}

