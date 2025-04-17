// compile

//go:build !386 && !arm && !mips && !mipsle && !amd64p32

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 6036: 6g's backend generates OINDREG with
// offsets larger than 32-bit.

package main

type T struct {
	Large [1 << 31]byte
	A     int
	B     int
}

func F(t *T) {
	t.B = t.A
}

type T2 [1<<31 + 2]byte

func F2(t *T2) {
	t[1<<31+1] = 42
}

type T3 [1<<15 + 1][1<<15 + 1]int

func F3(t *T3) {
	t[1<<15][1<<15] = 42
}

type S struct {
	A int32
	B int32
}

type T4 [1<<29 + 1]S

func F4(t *T4) {
	t[1<<29].B = 42
}
