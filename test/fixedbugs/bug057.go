// $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct {
	s string;
}


func main() {
	s := "";
	l1 := len(s);
	var t T;
	l2 := len(t.s);	// BUG: cannot take len() of a string field
	_, _ = l1, l2;
}

/*
uetli:/home/gri/go/test/bugs gri$ 6g bug057.go
bug057.go:14: syntax error
*/
