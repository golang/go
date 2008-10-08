// $G $D/$F.go && $L $F.$A && ! ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type S struct

type I interface {
	Foo()
}

func main() {
	var s *S;
	var i I;
	i = s;
}

// hide S down here to avoid static warning
type S struct {
}
