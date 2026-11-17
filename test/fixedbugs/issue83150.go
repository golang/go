// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 83150: on arm64, ppc64, riscv64 and s390x, a bounds panic whose
// index is a constant outside 0..31 and whose length is in a register
// reported the wrong index.

package main

import "strings"

var s = make([]int, 8)

func at(i, c int) int { return s[i-c] } // inlined: i-c folds to a constant

//go:noinline
func neg() int { return at(-1, 2) } // index -3

//go:noinline
func big() int { return s[40] }

//go:noinline
func slice() []int { return s[:40] }

func check(f func(), want string) {
	defer func() {
		r := recover()
		if r == nil {
			panic("no panic (bug)")
		}
		got := r.(error).Error()
		if !strings.Contains(got, want) {
			panic("got " + got + ", want " + want)
		}
	}()
	f()
}

func main() {
	check(func() { neg() }, "index out of range [-3]")
	check(func() { big() }, "index out of range [40] with length 8")
	check(func() { slice() }, "slice bounds out of range [:40] with capacity 8")
}
