// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var a = []int { 1, 2, }
var b = []int { }
var c = []int { 1 }

func main() {
	if len(a) != 2 { panicln("len a", len(a)) }
	if len(b) != 5 { panicln("len b", len(b)) }
	if len(c) != 1 { panicln("len a", len(c)) }
}
