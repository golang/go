// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	Foo: {
		return;
	}
	goto Foo;
}
/*
bug5.go:4: Foo undefined
bug5.go:4: fatal error: walktype: switch 1 unknown op GOTO l(4)
*/
