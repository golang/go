// run

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Was discarding function calls made for arguments named _
// in inlined functions.  Issue 3593.

package main

var did int

func main() {
	foo(side())
	foo2(side(), side())
	foo3(side(), side())
	T.m1(T(side()))
	T(1).m2(side())
	const want = 7
	if did != want {
		println("BUG: missing", want-did, "calls")
	}
}

func foo(_ int) {}
func foo2(_, _ int) {}
func foo3(int, int) {}
type T int
func (_ T) m1() {}
func (t T) m2(_ int) {}

func side() int {
	did++
	return 1
}
