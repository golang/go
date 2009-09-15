// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var s2 string = "\a\b\f\n\r\t\v";  // \r is miscompiled
	_ = s2;
}
/*
main.go.c: In function ‘main_main’:
main.go.c:20: error: missing terminating " character
main.go.c:21: error: missing terminating " character
main.go.c:24: error: ‘def’ undeclared (first use in this function)
main.go.c:24: error: (Each undeclared identifier is reported only once
main.go.c:24: error: for each function it appears in.)
main.go.c:24: error: syntax error before ‘def’
main.go.c:24: error: missing terminating " character
main.go.c:25: warning: excess elements in struct initializer
main.go.c:25: warning: (near initialization for ‘slit’)
main.go.c:36: error: syntax error at end of input
*/
