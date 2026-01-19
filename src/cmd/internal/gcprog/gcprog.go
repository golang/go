// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gcprog implements an encoder for packed GC pointer bitmaps,
// known as GC programs.
//
// # Program Format
//
// The GC program encodes a sequence of 0 and 1 bits indicating scalar or pointer words in an object.
// The encoding is a simple Lempel-Ziv program, with codes to emit literal bits and to repeat the
// last n bits c times.
//
// The possible codes are:
//
//	00000000: stop
//	0nnnnnnn: emit n bits copied from the next (n+7)/8 bytes, least significant bit first
//	10000000 n c: repeat the previous n bits c times; n, c are varints
//	1nnnnnnn c: repeat the previous n bits c times; c is a varint
//
// The numbers n and c, when they follow a code, are encoded as varints
// using the same encoding as encoding/binary's Uvarint.
package gcprog

import (
	"fmt"
	"io"
)

const progMaxLiteral = 127 // maximum n for literal n bit code

// A Writer is an encoder for GC programs.
//
// The typical use of a Writer is to call Init, maybe call Debug,
// make a sequence of Ptr, Advance, Repeat, and Append calls
// to describe the data type, and then finally call End.
type Writer struct {
	writeByte func(byte)
	index     int64
	b         [progMaxLiteral]byte
	nb        int
	debug     io.Writer
	debugBuf  []byte
}

// Init initializes w to write a new GC program
// by calling writeByte for each byte in the program.
func (w *Writer) Init(writeByte func(byte)) {
	w.writeByte = writeByte
}

// Debug causes the writer to print a debugging trace to out
// during future calls to methods like Ptr, Advance, and End.
// It also enables debugging checks during the encoding.
func (w *Writer) Debug(out io.Writer) {
	w.debug = out
}

// byte writes the byte x to the output.
func (w *Writer) byte(x byte) {
	if w.debug != nil {
		w.debugBuf = append(w.debugBuf, x)
	}
	w.writeByte(x)
}

// End marks the end of the program, writing any remaining bytes.
func (w *Writer) End() {
	w.flushlit()
	w.byte(0)
	if w.debug != nil {
		index := progbits(w.debugBuf)
		if index != w.index {
			println("gcprog: End wrote program for", index, "bits, but current index is", w.index)
			panic("gcprog: out of sync")
		}
	}
}

// Ptr emits a 1 into the bit stream at the given bit index.
// that is, it records that the index'th word in the object memory is a pointer.
// Any bits between the current index and the new index
// are set to zero, meaning the corresponding words are scalars.
func (w *Writer) Ptr(index int64) {
	if index < w.index {
		println("gcprog: Ptr at index", index, "but current index is", w.index)
		panic("gcprog: invalid Ptr index")
	}
	w.ZeroUntil(index)
	if w.debug != nil {
		fmt.Fprintf(w.debug, "gcprog: ptr at %d\n", index)
	}
	w.lit(1)
}

// Repeat emits an instruction to repeat the description
// of the last n words c times (including the initial description, c+1 times in total).
func (w *Writer) Repeat(n, c int64) {
	if n == 0 || c == 0 {
		return
	}
	w.flushlit()
	if w.debug != nil {
		fmt.Fprintf(w.debug, "gcprog: repeat %d × %d\n", n, c)
	}
	if n < 128 {
		w.byte(0x80 | byte(n))
	} else {
		w.byte(0x80)
		w.varint(n)
	}
	w.varint(c)
	w.index += n * c
}

// ZeroUntil adds zeros to the bit stream until reaching the given index;
// that is, it records that the words from the most recent pointer until
// the index'th word are scalars.
// ZeroUntil is usually called in preparation for a call to Repeat, Append, or End.
func (w *Writer) ZeroUntil(index int64) {
	if index < w.index {
		println("gcprog: Advance", index, "but index is", w.index)
		panic("gcprog: invalid Advance index")
	}
	skip := (index - w.index)
	if skip == 0 {
		return
	}
	if skip < 4*8 {
		if w.debug != nil {
			fmt.Fprintf(w.debug, "gcprog: advance to %d by literals\n", index)
		}
		for i := int64(0); i < skip; i++ {
			w.lit(0)
		}
		return
	}

	if w.debug != nil {
		fmt.Fprintf(w.debug, "gcprog: advance to %d by repeat\n", index)
	}
	w.lit(0)
	w.flushlit()
	w.Repeat(1, skip-1)
}

// progbits returns the length of the bit stream encoded by the program p.
func progbits(p []byte) int64 {
	var n int64
	for len(p) > 0 {
		x := p[0]
		p = p[1:]
		if x == 0 {
			break
		}
		if x&0x80 == 0 {
			count := x &^ 0x80
			n += int64(count)
			p = p[(count+7)/8:]
			continue
		}
		nbit := int64(x &^ 0x80)
		if nbit == 0 {
			nbit, p = readvarint(p)
		}
		var count int64
		count, p = readvarint(p)
		n += nbit * count
	}
	if len(p) > 0 {
		println("gcprog: found end instruction after", n, "ptrs, with", len(p), "bytes remaining")
		panic("gcprog: extra data at end of program")
	}
	return n
}

// readvarint reads a varint from p, returning the value and the remainder of p.
func readvarint(p []byte) (int64, []byte) {
	var v int64
	var nb uint
	for {
		c := p[0]
		p = p[1:]
		v |= int64(c&^0x80) << nb
		nb += 7
		if c&0x80 == 0 {
			break
		}
	}
	return v, p
}

// lit adds a single literal bit to w.
func (w *Writer) lit(x byte) {
	if w.nb == progMaxLiteral {
		w.flushlit()
	}
	w.b[w.nb] = x
	w.nb++
	w.index++
}

// varint emits the varint encoding of x.
func (w *Writer) varint(x int64) {
	if x < 0 {
		panic("gcprog: negative varint")
	}
	for x >= 0x80 {
		w.byte(byte(0x80 | x))
		x >>= 7
	}
	w.byte(byte(x))
}

// flushlit flushes any pending literal bits.
func (w *Writer) flushlit() {
	if w.nb == 0 {
		return
	}
	if w.debug != nil {
		fmt.Fprintf(w.debug, "gcprog: flush %d literals\n", w.nb)
		fmt.Fprintf(w.debug, "\t%v\n", w.b[:w.nb])
		fmt.Fprintf(w.debug, "\t%02x", byte(w.nb))
	}
	w.byte(byte(w.nb))
	var bits uint8
	for i := 0; i < w.nb; i++ {
		bits |= w.b[i] << uint(i%8)
		if (i+1)%8 == 0 {
			if w.debug != nil {
				fmt.Fprintf(w.debug, " %02x", bits)
			}
			w.byte(bits)
			bits = 0
		}
	}
	if w.nb%8 != 0 {
		if w.debug != nil {
			fmt.Fprintf(w.debug, " %02x", bits)
		}
		w.byte(bits)
	}
	if w.debug != nil {
		fmt.Fprintf(w.debug, "\n")
	}
	w.nb = 0
}
