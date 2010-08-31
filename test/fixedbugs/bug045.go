// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct {
	i int
}

func main() {
	var ta []*T;

	ta = new([1]*T)[0:];
	ta[0] = nil;
}
/*
bug045.go:13: fatal error: goc: exit 1
*/
