// compile

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// PR61253: gccgo incorrectly parsed the
// `RecvStmt = ExpressionList "=" RecvExpr` production.

package main

func main() {
	c := make(chan int)
	v := new(int)
	b := new(bool)
	select {
	case (*v), (*b) = <-c:
	}

}
