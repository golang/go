// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "testing"

type extTest struct {
	f    func(uint64, uint64) uint64
	arg1 uint64
	arg2 uint64
	res  uint64
	name string
}

var extTests = [...]extTest{
	{f: func(a, b uint64) uint64 { op1 := int32(a); op2 := int32(b); return uint64(uint32(op1 / op2)) }, arg1: 0x1, arg2: 0xfffffffeffffffff, res: 0xffffffff, name: "div"},
	{f: func(a, b uint64) uint64 { op1 := int32(a); op2 := int32(b); return uint64(uint32(op1 * op2)) }, arg1: 0x1, arg2: 0x100000001, res: 0x1, name: "mul"},
	{f: func(a, b uint64) uint64 { op1 := int32(a); op2 := int32(b); return uint64(uint32(op1 + op2)) }, arg1: 0x1, arg2: 0xeeeeeeeeffffffff, res: 0x0, name: "add"},
	{f: func(a, b uint64) uint64 { op1 := int32(a); op2 := int32(b); return uint64(uint32(op1 - op2)) }, arg1: 0x1, arg2: 0xeeeeeeeeffffffff, res: 0x2, name: "sub"},
	{f: func(a, b uint64) uint64 { op1 := int32(a); op2 := int32(b); return uint64(uint32(op1 | op2)) }, arg1: 0x100000000000001, arg2: 0xfffffffffffffff, res: 0xffffffff, name: "or"},
	{f: func(a, b uint64) uint64 { op1 := int32(a); op2 := int32(b); return uint64(uint32(op1 ^ op2)) }, arg1: 0x100000000000001, arg2: 0xfffffffffffffff, res: 0xfffffffe, name: "xor"},
	{f: func(a, b uint64) uint64 { op1 := int32(a); op2 := int32(b); return uint64(uint32(op1 & op2)) }, arg1: 0x100000000000001, arg2: 0x100000000000001, res: 0x1, name: "and"},
}

func TestZeroExtension(t *testing.T) {
	for _, x := range extTests {
		r := x.f(x.arg1, x.arg2)
		if x.res != r {
			t.Errorf("%s: got %d want %d", x.name, r, x.res)
		}
	}
}
