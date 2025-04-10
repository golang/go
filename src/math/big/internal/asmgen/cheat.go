// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

// This program can be compiled with -S to produce a “cheat sheet”
// for filling out a new Arch: the compiler will show you how to implement
// the various operations.
//
// Usage (replace TARGET with your target architecture):
//
//	GOOS=linux GOARCH=TARGET go build -gcflags='-p=cheat -S' cheat.go

package p

import "math/bits"

func mov(x, y uint) uint             { return y }
func zero() uint                     { return 0 }
func add(x, y uint) uint             { return x + y }
func adds(x, y, c uint) (uint, uint) { return bits.Add(x, y, 0) }
func adcs(x, y, c uint) (uint, uint) { return bits.Add(x, y, c) }
func sub(x, y uint) uint             { return x + y }
func subs(x, y uint) (uint, uint)    { return bits.Sub(x, y, 0) }
func sbcs(x, y, c uint) (uint, uint) { return bits.Sub(x, y, c) }
func mul(x, y uint) uint             { return x * y }
func mulWide(x, y uint) (uint, uint) { return bits.Mul(x, y) }
func lsh(x, s uint) uint             { return x << s }
func rsh(x, s uint) uint             { return x >> s }
func and(x, y uint) uint             { return x & y }
func or(x, y uint) uint              { return x | y }
func xor(x, y uint) uint             { return x ^ y }
func neg(x uint) uint                { return -x }
func loop(x int) int {
	s := 0
	for i := 1; i < x; i++ {
		s += i
		if s == 98 { // useful for jmpEqual
			return 99
		}
		if s == 99 {
			return 100
		}
		if s == 0 { // useful for jmpZero
			return 101
		}
		if s != 0 { // useful for jmpNonZero
			s *= 3
		}
		s += 2 // keep last condition from being inverted
	}
	return s
}
func mem(x *[10]struct{ a, b uint }, i int) uint { return x[i].b }
