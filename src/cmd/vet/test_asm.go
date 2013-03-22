// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// This file contains declarations to test the assembly in test_asm.s.

package main

func arg1(x int8, y uint8)
func arg2(x int16, y uint16)
func arg4(x int32, y uint32)
func arg8(x int64, y uint64)
func argint(x int, y uint)
func argptr(x *byte, y *byte, c chan int, m map[int]int, f func())
func argstring(x, y string)
func argslice(x, y []string)
func argiface(x interface{}, y interface {
	m()
})
func returnint() int
func returnbyte(x int) byte
func returnnamed(x byte) (r1 int, r2 int16, r3 string, r4 byte)
