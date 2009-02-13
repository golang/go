// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var a = []int ( 1,2, )
var b = [5]int ( 1,2,3 )
var c = []int ( 1 )
var d = [...]int ( 1,2,3 )

func main() {
	if len(a) != 2 { panicln("len a", len(a)) }
	if len(b) != 5 { panicln("len b", len(b)) }
	if len(c) != 1 { panicln("len d", len(c)) }
	if len(d) != 3 { panicln("len c", len(d)) }

	if a[0] != 1 { panicln("a[0]", a[0]) }
	if a[1] != 2 { panicln("a[1]", a[1]) }

	if b[0] != 1 { panicln("b[0]", b[0]) }
	if b[1] != 2 { panicln("b[1]", b[1]) }
	if b[2] != 3 { panicln("b[2]", b[2]) }
	if b[3] != 0 { panicln("b[3]", b[3]) }
	if b[4] != 0 { panicln("b[4]", b[4]) }

	if c[0] != 1 { panicln("c[0]", c[0]) }

	if d[0] != 1 { panicln("d[0]", d[0]) }
	if d[1] != 2 { panicln("d[1]", d[1]) }
	if d[2] != 3 { panicln("d[2]", d[2]) }
}
