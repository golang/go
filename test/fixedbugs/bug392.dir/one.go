// Copyright 2012 The Go Authors.  All rights reserved.
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
