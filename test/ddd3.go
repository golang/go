// $G $D/ddd2.go && $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./ddd2"

func main() {
	if x := ddd.Sum(1, 2, 3); x != 6 {
		panicln("ddd.Sum 6", x)
	}
	if x := ddd.Sum(); x != 0 {
		panicln("ddd.Sum 0", x)
	}
	if x := ddd.Sum(10); x != 10 {
		panicln("ddd.Sum 10", x)
	}
	if x := ddd.Sum(1, 8); x != 9 {
		panicln("ddd.Sum 9", x)
	}
}
