// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $D/$F.go && $L $F.$A && ./$A.out

package main

func main() {
	// This bug doesn't arise with [...]int, or []interface{} or [3]interface{}.
	a := [...]interface{} { 1, 2, 3 };
	n := 0;
	for _, v := range a {
		if v.(int) != n {
			panicln("BUG:", n, v.(int));
		}
		n++;
	}
}
