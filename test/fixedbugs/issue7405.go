// compile

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 7405: the equality function for struct with many
// embedded fields became more complex after fixing issue 7366,
// leading to out of registers on 386.

package p

type T1 struct {
	T2
	T3
	T4
}

type T2 struct {
	Conn
}

type T3 struct {
	PacketConn
}

type T4 struct {
	PacketConn
	T5
}

type T5 struct {
	x int
	T6
}

type T6 struct {
	y, z int
}

type Conn interface {
	A()
}

type PacketConn interface {
	B()
}

func F(a, b T1) bool {
	return a == b
}
