// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct {
	i int
}

type IN interface {
}

func main() {
	var i *int;
	var f *float;
	var s *string;
	var m *map[float] *int;
	var c *chan int;
	var t *T;
	var in IN;
	var ta []IN;

	i = nil;
	f = nil;
	s = nil;
	m = nil;
	c = nil;
	t = nil;
	i = nil;
	ta = new([1]IN);
	ta[0] = nil;
}
