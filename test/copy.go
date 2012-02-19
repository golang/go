// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Semi-exhaustive test for the copy predeclared function.

package main

import (
	"fmt"
	"os"
)

const N = 40

var input8 = make([]uint8, N)
var output8 = make([]uint8, N)
var input16 = make([]uint16, N)
var output16 = make([]uint16, N)
var input32 = make([]uint32, N)
var output32 = make([]uint32, N)
var input64 = make([]uint64, N)
var output64 = make([]uint64, N)
var inputS string
var outputS = make([]uint8, N)

type my8 []uint8
type my16 []uint16
type my32 []uint32
type my32b []uint32
type my64 []uint64
type myS string

func u8(i int) uint8 {
	i = 'a' + i%26
	return uint8(i)
}

func u16(ii int) uint16 {
	var i = uint16(ii)
	i = 'a' + i%26
	i |= i << 8
	return i
}

func u32(ii int) uint32 {
	var i = uint32(ii)
	i = 'a' + i%26
	i |= i << 8
	i |= i << 16
	return i
}

func u64(ii int) uint64 {
	var i = uint64(ii)
	i = 'a' + i%26
	i |= i << 8
	i |= i << 16
	i |= i << 32
	return i
}

func reset() {
	// swap in and out to exercise copy-up and copy-down
	input8, output8 = output8, input8
	input16, output16 = output16, input16
	input32, output32 = output32, input32
	input64, output64 = output64, input64
	in := 0
	out := 13
	for i := range input8 {
		input8[i] = u8(in)
		output8[i] = u8(out)
		outputS[i] = u8(out)
		input16[i] = u16(in)
		output16[i] = u16(out)
		input32[i] = u32(in)
		output32[i] = u32(out)
		input64[i] = u64(in)
		output64[i] = u64(out)
		in++
		out++
	}
	inputS = string(input8)
}

func clamp(n int) int {
	if n > N {
		return N
	}
	return n
}

func ncopied(length, in, out int) int {
	n := length
	if in+n > N {
		n = N - in
	}
	if out+n > N {
		n = N - out
	}
	return n
}

func doAllSlices(length, in, out int) {
	reset()
	n := copy(my8(output8[out:clamp(out+length)]), input8[in:clamp(in+length)])
	verify8(length, in, out, n)
	n = copy(my8(outputS[out:clamp(out+length)]), myS(inputS[in:clamp(in+length)]))
	verifyS(length, in, out, n)
	n = copy(my16(output16[out:clamp(out+length)]), input16[in:clamp(in+length)])
	verify16(length, in, out, n)
	n = copy(my32(output32[out:clamp(out+length)]), my32b(input32[in:clamp(in+length)]))
	verify32(length, in, out, n)
	n = copy(my64(output64[out:clamp(out+length)]), input64[in:clamp(in+length)])
	verify64(length, in, out, n)
}

func bad8(state string, i, length, in, out int) {
	fmt.Printf("%s bad(%d %d %d): %c not %c:\n\t%s\n\t%s\n",
		state,
		length, in, out,
		output8[i],
		uint8(i+13),
		input8, output8)
	os.Exit(1)
}

func verify8(length, in, out, m int) {
	n := ncopied(length, in, out)
	if m != n {
		fmt.Printf("count bad(%d %d %d): %d not %d\n", length, in, out, m, n)
		return
	}
	// before
	var i int
	for i = 0; i < out; i++ {
		if output8[i] != u8(i+13) {
			bad8("before8", i, length, in, out)
			return
		}
	}
	// copied part
	for ; i < out+n; i++ {
		if output8[i] != u8(i+in-out) {
			bad8("copied8", i, length, in, out)
			return
		}
	}
	// after
	for ; i < len(output8); i++ {
		if output8[i] != u8(i+13) {
			bad8("after8", i, length, in, out)
			return
		}
	}
}

func badS(state string, i, length, in, out int) {
	fmt.Printf("%s bad(%d %d %d): %c not %c:\n\t%s\n\t%s\n",
		state,
		length, in, out,
		outputS[i],
		uint8(i+13),
		inputS, outputS)
	os.Exit(1)
}

func verifyS(length, in, out, m int) {
	n := ncopied(length, in, out)
	if m != n {
		fmt.Printf("count bad(%d %d %d): %d not %d\n", length, in, out, m, n)
		return
	}
	// before
	var i int
	for i = 0; i < out; i++ {
		if outputS[i] != u8(i+13) {
			badS("beforeS", i, length, in, out)
			return
		}
	}
	// copied part
	for ; i < out+n; i++ {
		if outputS[i] != u8(i+in-out) {
			badS("copiedS", i, length, in, out)
			return
		}
	}
	// after
	for ; i < len(outputS); i++ {
		if outputS[i] != u8(i+13) {
			badS("afterS", i, length, in, out)
			return
		}
	}
}

func bad16(state string, i, length, in, out int) {
	fmt.Printf("%s bad(%d %d %d): %x not %x:\n\t%v\n\t%v\n",
		state,
		length, in, out,
		output16[i],
		uint16(i+13),
		input16, output16)
	os.Exit(1)
}

func verify16(length, in, out, m int) {
	n := ncopied(length, in, out)
	if m != n {
		fmt.Printf("count bad(%d %d %d): %d not %d\n", length, in, out, m, n)
		return
	}
	// before
	var i int
	for i = 0; i < out; i++ {
		if output16[i] != u16(i+13) {
			bad16("before16", i, length, in, out)
			return
		}
	}
	// copied part
	for ; i < out+n; i++ {
		if output16[i] != u16(i+in-out) {
			bad16("copied16", i, length, in, out)
			return
		}
	}
	// after
	for ; i < len(output16); i++ {
		if output16[i] != u16(i+13) {
			bad16("after16", i, length, in, out)
			return
		}
	}
}

func bad32(state string, i, length, in, out int) {
	fmt.Printf("%s bad(%d %d %d): %x not %x:\n\t%v\n\t%v\n",
		state,
		length, in, out,
		output32[i],
		uint32(i+13),
		input32, output32)
	os.Exit(1)
}

func verify32(length, in, out, m int) {
	n := ncopied(length, in, out)
	if m != n {
		fmt.Printf("count bad(%d %d %d): %d not %d\n", length, in, out, m, n)
		return
	}
	// before
	var i int
	for i = 0; i < out; i++ {
		if output32[i] != u32(i+13) {
			bad32("before32", i, length, in, out)
			return
		}
	}
	// copied part
	for ; i < out+n; i++ {
		if output32[i] != u32(i+in-out) {
			bad32("copied32", i, length, in, out)
			return
		}
	}
	// after
	for ; i < len(output32); i++ {
		if output32[i] != u32(i+13) {
			bad32("after32", i, length, in, out)
			return
		}
	}
}

func bad64(state string, i, length, in, out int) {
	fmt.Printf("%s bad(%d %d %d): %x not %x:\n\t%v\n\t%v\n",
		state,
		length, in, out,
		output64[i],
		uint64(i+13),
		input64, output64)
	os.Exit(1)
}

func verify64(length, in, out, m int) {
	n := ncopied(length, in, out)
	if m != n {
		fmt.Printf("count bad(%d %d %d): %d not %d\n", length, in, out, m, n)
		return
	}
	// before
	var i int
	for i = 0; i < out; i++ {
		if output64[i] != u64(i+13) {
			bad64("before64", i, length, in, out)
			return
		}
	}
	// copied part
	for ; i < out+n; i++ {
		if output64[i] != u64(i+in-out) {
			bad64("copied64", i, length, in, out)
			return
		}
	}
	// after
	for ; i < len(output64); i++ {
		if output64[i] != u64(i+13) {
			bad64("after64", i, length, in, out)
			return
		}
	}
}

func slice() {
	for length := 0; length < N; length++ {
		for in := 0; in <= 32; in++ {
			for out := 0; out <= 32; out++ {
				doAllSlices(length, in, out)
			}
		}
	}
}

// Array test. Can be much simpler. It's only checking for correct handling of [0:].
func array() {
	var array [N]uint8
	reset()
	copy(array[0:], input8)
	for i := 0; i < N; i++ {
		output8[i] = 0
	}
	copy(output8, array[0:])
	verify8(N, 0, 0, N)
}

func main() {
	slice()
	array()
}
