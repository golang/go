// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG should not crash

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Interface
type I interface { F() int }

// Implements interface
type S struct { }
func (s *S) F() int { return 1 }

// Allocates S but returns I
// Arg is unused but important:
// if you take it out (and the 0s below)
// then the bug goes away.
func NewI(i int) I {
	return new(S)
}

// Uses interface method.
func Use(x I) {
	x.F()
}

func main() {
	i := NewI(0);
	Use(i);

	// Again, without temporary
	// Crashes because x.F is 0.
	Use(NewI(0));
}

