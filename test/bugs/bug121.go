// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T ()

type I interface {
	f, g ();
	h T;  // should only allow FunctionType here
}

type S struct {
}

func (s *S) f() {}
func (s *S) g() {}
func (s *S) h() {}  // here we can't write (s *S) T either

func main() {
	var i I = new(*S);
}
