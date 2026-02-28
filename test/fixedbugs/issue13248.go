// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This program caused an infinite loop with the recursive-descent parser.

package main

func main() {
    foo( // GCCGO_ERROR "undefined name"
} // ERROR "unexpected }|expected operand|missing"
