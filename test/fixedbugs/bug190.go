// compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type S struct {
	p *S;
	s []S;
	m map[int] S;
	c chan S;
	i interface { f(S); };
	f func(S) S;
}

func main() {
	var s S;
	s.p = &s;
	s.s = make([]S, 1);
	s.s[0] = s;
	s.m[0] = s;
	s.c <- s;
	s.i.f(s);
}
