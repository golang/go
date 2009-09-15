// $G $D/$F.go || echo BUG should compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const c = 1;

func main() {
	c := 0;
	_ = c;
}

/*
bug144.go:8: left side of := must be a name
bug144.go:8: operation LITERAL not allowed in assignment context
bug144.go:8: illegal types for operand: AS
	ideal
	int
*/
