// run

//go:build amd64 && (linux || darwin)

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"syscall"
)

// Use global variables so the compiler
// doesn't know that they are constants.
var p = syscall.Getpagesize()
var zero = 0
var one = 1

func main() {
	// Allocate 2 pages of memory.
	b, err := syscall.Mmap(-1, 0, 2*p, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		panic(err)
	}
	// Mark the second page as faulting.
	err = syscall.Mprotect(b[p:], syscall.PROT_NONE)
	if err != nil {
		panic(err)
	}
	// Get a slice pointing to the last byte of the good page.
	x := b[p-one : p]

	test16(x)
	test16i(x, 0)
	test32(x)
	test32i(x, 0)
	test64(x)
	test64i(x, 0)
}

func test16(x []byte) uint16 {
	defer func() {
		r := recover()
		if r == nil {
			panic("no fault or bounds check failure happened")
		}
		s := fmt.Sprintf("%s", r)
		if s != "runtime error: index out of range [1] with length 1" {
			panic("bad panic: " + s)
		}
	}()
	// Try to read 2 bytes from x.
	return uint16(x[0]) | uint16(x[1])<<8

	// We expect to get an "index out of range" error from x[1].
	// If we promote the first load to a 2-byte load, it will segfault, which we don't want.
}

func test16i(x []byte, i int) uint16 {
	defer func() {
		r := recover()
		if r == nil {
			panic("no fault or bounds check failure happened")
		}
		s := fmt.Sprintf("%s", r)
		if s != "runtime error: index out of range [1] with length 1" {
			panic("bad panic: " + s)
		}
	}()
	return uint16(x[i]) | uint16(x[i+1])<<8
}

func test32(x []byte) uint32 {
	defer func() {
		r := recover()
		if r == nil {
			panic("no fault or bounds check failure happened")
		}
		s := fmt.Sprintf("%s", r)
		if s != "runtime error: index out of range [1] with length 1" {
			panic("bad panic: " + s)
		}
	}()
	return uint32(x[0]) | uint32(x[1])<<8 | uint32(x[2])<<16 | uint32(x[3])<<24
}

func test32i(x []byte, i int) uint32 {
	defer func() {
		r := recover()
		if r == nil {
			panic("no fault or bounds check failure happened")
		}
		s := fmt.Sprintf("%s", r)
		if s != "runtime error: index out of range [1] with length 1" {
			panic("bad panic: " + s)
		}
	}()
	return uint32(x[i]) | uint32(x[i+1])<<8 | uint32(x[i+2])<<16 | uint32(x[i+3])<<24
}

func test64(x []byte) uint64 {
	defer func() {
		r := recover()
		if r == nil {
			panic("no fault or bounds check failure happened")
		}
		s := fmt.Sprintf("%s", r)
		if s != "runtime error: index out of range [1] with length 1" {
			panic("bad panic: " + s)
		}
	}()
	return uint64(x[0]) | uint64(x[1])<<8 | uint64(x[2])<<16 | uint64(x[3])<<24 |
		uint64(x[4])<<32 | uint64(x[5])<<40 | uint64(x[6])<<48 | uint64(x[7])<<56
}

func test64i(x []byte, i int) uint64 {
	defer func() {
		r := recover()
		if r == nil {
			panic("no fault or bounds check failure happened")
		}
		s := fmt.Sprintf("%s", r)
		if s != "runtime error: index out of range [1] with length 1" {
			panic("bad panic: " + s)
		}
	}()
	return uint64(x[i+0]) | uint64(x[i+1])<<8 | uint64(x[i+2])<<16 | uint64(x[i+3])<<24 |
		uint64(x[i+4])<<32 | uint64(x[i+5])<<40 | uint64(x[i+6])<<48 | uint64(x[i+7])<<56
}
