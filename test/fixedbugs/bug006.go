// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const (
	x float = iota;
	g float = 4.5 * iota;
);

func main() {
	if g == 0.0 { print("zero\n");}
	if g != 4.5 { print(" fail\n"); sys.Exit(1); }
}
