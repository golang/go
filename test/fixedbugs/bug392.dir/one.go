// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Functions that the inliner exported incorrectly.

package one

type T int

// Issue 2678
func F1(T *T) bool { return T == nil }

// Issue 2682.
func F2(c chan int) bool { return c == (<-chan int)(nil) }

// Use of single named return value.
func F3() (ret []int) { return append(ret, 1) }

// Call of inlined method with blank receiver.
func (_ *T) M() int { return 1 }
func (t *T) MM() int { return t.M() }


// One more like issue 2678
type S struct { x, y int }
type U []S

func F4(S int) U { return U{{S,S}} }

func F5() []*S {
	return []*S{ {1,2}, { 3, 4} }
}

func F6(S int) *U {
	return &U{{S,S}}
}

// Bug in the fix.

type PB struct { x int }

func (t *PB) Reset() { *t = PB{} }
