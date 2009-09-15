// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var i int;
	i = '\'';
	i = '\\';
	var s string;
	s = "\"";
	_, _ = i, s;
}
/*
bug.go:5: unknown escape sequence: '
bug.go:6: unknown escape sequence: \
bug.go:8: unknown escape sequence: "
*/
