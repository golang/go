// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// http://code.google.com/p/go/issues/detail?id=589

package main

import "unsafe"

var bug = false

var minus1 = -1
var big int64 = 10 | 1<<32

var g1 []int

func shouldfail(f func(), desc string) {
	defer func() { recover() }()
	f()
	if !bug {
		println("BUG")
		bug = true
	}
	println("didn't crash: ", desc)
}

func badlen() {
	g1 = make([]int, minus1)
}

func biglen() {
	g1 = make([]int, big)
}

func badcap() {
	g1 = make([]int, 10, minus1)
}

func badcap1() {
	g1 = make([]int, 10, 5)
}

func bigcap() {
	g1 = make([]int, 10, big)
}

var g3 map[int]int
func badmapcap() {
	g3 = make(map[int]int, minus1)
}

func bigmapcap() {
	g3 = make(map[int]int, big)
}

var g4 chan int
func badchancap() {
	g4 = make(chan int, minus1)
}

func bigchancap() {
	g4 = make(chan int, big)
}

const addrBits = unsafe.Sizeof((*byte)(nil))

var g5 chan [1<<15]byte
func overflowchan() {
	if addrBits == 32 {
		g5 = make(chan [1<<15]byte, 1<<20)
	} else {
		// cannot overflow on 64-bit, because
		// int is 32 bits and max chan value size
		// in the implementation is 64 kB.
		panic(1)
	}
}

func main() {
	shouldfail(badlen, "badlen")
	shouldfail(biglen, "biglen")
	shouldfail(badcap, "badcap")
	shouldfail(badcap1, "badcap1")
	shouldfail(bigcap, "bigcap")
	shouldfail(badmapcap, "badmapcap")
	shouldfail(bigmapcap, "bigmapcap")
	shouldfail(badchancap, "badchancap")
	shouldfail(bigchancap, "bigchancap")
	shouldfail(overflowchan, "overflowchan")
}
