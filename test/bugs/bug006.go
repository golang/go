// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const (
	g float = 4.5 * iota;
);

func main() {
}
/*
should 4.5 * iota be ok? perhaps, perhaps not. but (all!) error msgs are bad:
bug6.go:4: illegal combination of literals 0 0
bug6.go:4: expression must be a constant
bug6.go:4: expression must be a constant
bug6.go:4: expression must be a constant
bug6.go:4: expression must be a constant
bug6.go:4: expression must be a constant
bug6.go:4: expression must be a constant
bug6.go:4: expression must be a constant
bug6.go:4: expression must be a constant
bug6.go:4: expression must be a constant
bug6.go:4: fatal error: too many errors
*/
