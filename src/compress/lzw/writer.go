// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzw

import (
	"bufio"
	"errors"
	"fmt"
	"io"
)

// A writer is a buffered, flushable writer.
type writer interface {
	io.ByteWriter
	Flush() error
}

const (
	// A code is a 12 bit value, stored as a uint32 when encoding to avoid
	// type conversions when shifting bits.
	maxCode     = 1<<12 - 1
	invalidCode = 1<<32 - 1
	// There are 1<<12 possible codes, which is an upper bound on the number of
	// valid hash table entries at any given point in time. tableSize is 4x that.
	tableSize = 4 * 1 << 12
	tableMask = tableSize - 1
	// A hash table entry is a uint32. Zero is an invalid entry since the
	// lower 12 bits of a valid entry must be a non-literal code.
	invalidEntry = 0
)

// Writer is an LZW compressor. It writes the compressed form of the data
// to an underlying writer (see [NewWriter]).
type Writer struct {
	// w is the writer that compressed bytes are written to.
	w writer
	// litWidth is the width in bits of literal codes.
	litWidth uint
	// order, write, bits, nBits and width are the state for
	// converting a code stream into a byte stream.
	order Order
	write func(*Writer, uint32) error
	nBits uint
	width uint
	bits  uint32
	// hi is the code implied by the next code emission.
	// overflow is the code at which hi overflows the code width.
	hi, overflow uint32
	// savedCode is the accumulated code at the end of the most recent Write
	// call. It is equal to invalidCode if there was no such call.
	savedCode uint32
	// err is the first error encountered during writing. Closing the writer
	// will make any future Write calls return errClosed
	err error
	// table is the hash table from 20-bit keys to 12-bit values. Each table
	// entry contains key<<12|val and collisions resolve by linear probing.
	// The keys consist of a 12-bit code prefix and an 8-bit byte suffix.
	// The values are a 12-bit code.
	table [tableSize]uint32
}

// writeLSB writes the code c for "Least Significant Bits first" data.
func (w *Writer) writeLSB(c uint32) error {
	w.bits |= c << w.nBits
	w.nBits += w.width
	for w.nBits >= 8 {
		if err := w.w.WriteByte(uint8(w.bits)); err != nil {
			return err
		}
		w.bits >>= 8
		w.nBits -= 8
	}
	return nil
}

// writeMSB writes the code c for "Most Significant Bits first" data.
func (w *Writer) writeMSB(c uint32) error {
	w.bits |= c << (32 - w.width - w.nBits)
	w.nBits += w.width
	for w.nBits >= 8 {
		if err := w.w.WriteByte(uint8(w.bits >> 24)); err != nil {
			return err
		}
		w.bits <<= 8
		w.nBits -= 8
	}
	return nil
}

// errOutOfCodes is an internal error that means that the writer has run out
// of unused codes and a clear code needs to be sent next.
var errOutOfCodes = errors.New("lzw: out of codes")

// incHi increments e.hi and checks for both overflow and running out of
// unused codes. In the latter case, incHi sends a clear code, resets the
// writer state and returns errOutOfCodes.
func (w *Writer) incHi() error {
	w.hi++
	if w.hi == w.overflow {
		w.width++
		w.overflow <<= 1
	}
	if w.hi == maxCode {
		clear := uint32(1) << w.litWidth
		if err := w.write(w, clear); err != nil {
			return err
		}
		w.width = w.litWidth + 1
		w.hi = clear + 1
		w.overflow = clear << 1
		for i := range w.table {
			w.table[i] = invalidEntry
		}
		return errOutOfCodes
	}
	return nil
}

// Write writes a compressed representation of p to w's underlying writer.
func (w *Writer) Write(p []byte) (n int, err error) {
	if w.err != nil {
		return 0, w.err
	}
	if len(p) == 0 {
		return 0, nil
	}
	if maxLit := uint8(1<<w.litWidth - 1); maxLit != 0xff {
		for _, x := range p {
			if x > maxLit {
				w.err = errors.New("lzw: input byte too large for the litWidth")
				return 0, w.err
			}
		}
	}
	n = len(p)
	code := w.savedCode
	if code == invalidCode {
		// This is the first write; send a clear code.
		// https://www.w3.org/Graphics/GIF/spec-gif89a.txt Appendix F
		// "Variable-Length-Code LZW Compression" says that "Encoders should
		// output a Clear code as the first code of each image data stream".
		//
		// LZW compression isn't only used by GIF, but it's cheap to follow
		// that directive unconditionally.
		clear := uint32(1) << w.litWidth
		if err := w.write(w, clear); err != nil {
			return 0, err
		}
		// After the starting clear code, the next code sent (for non-empty
		// input) is always a literal code.
		code, p = uint32(p[0]), p[1:]
	}
loop:
	for _, x := range p {
		literal := uint32(x)
		key := code<<8 | literal
		// If there is a hash table hit for this key then we continue the loop
		// and do not emit a code yet.
		hash := (key>>12 ^ key) & tableMask
		for h, t := hash, w.table[hash]; t != invalidEntry; {
			if key == t>>12 {
				code = t & maxCode
				continue loop
			}
			h = (h + 1) & tableMask
			t = w.table[h]
		}
		// Otherwise, write the current code, and literal becomes the start of
		// the next emitted code.
		if w.err = w.write(w, code); w.err != nil {
			return 0, w.err
		}
		code = literal
		// Increment e.hi, the next implied code. If we run out of codes, reset
		// the writer state (including clearing the hash table) and continue.
		if err1 := w.incHi(); err1 != nil {
			if err1 == errOutOfCodes {
				continue
			}
			w.err = err1
			return 0, w.err
		}
		// Otherwise, insert key -> e.hi into the map that e.table represents.
		for {
			if w.table[hash] == invalidEntry {
				w.table[hash] = (key << 12) | w.hi
				break
			}
			hash = (hash + 1) & tableMask
		}
	}
	w.savedCode = code
	return n, nil
}

// Close closes the [Writer], flushing any pending output. It does not close
// w's underlying writer.
func (w *Writer) Close() error {
	if w.err != nil {
		if w.err == errClosed {
			return nil
		}
		return w.err
	}
	// Make any future calls to Write return errClosed.
	w.err = errClosed
	// Write the savedCode if valid.
	if w.savedCode != invalidCode {
		if err := w.write(w, w.savedCode); err != nil {
			return err
		}
		if err := w.incHi(); err != nil && err != errOutOfCodes {
			return err
		}
	} else {
		// Write the starting clear code, as w.Write did not.
		clear := uint32(1) << w.litWidth
		if err := w.write(w, clear); err != nil {
			return err
		}
	}
	// Write the eof code.
	eof := uint32(1)<<w.litWidth + 1
	if err := w.write(w, eof); err != nil {
		return err
	}
	// Write the final bits.
	if w.nBits > 0 {
		if w.order == MSB {
			w.bits >>= 24
		}
		if err := w.w.WriteByte(uint8(w.bits)); err != nil {
			return err
		}
	}
	return w.w.Flush()
}

// Reset clears the [Writer]'s state and allows it to be reused again
// as a new [Writer].
func (w *Writer) Reset(dst io.Writer, order Order, litWidth int) {
	*w = Writer{}
	w.init(dst, order, litWidth)
}

// NewWriter creates a new [io.WriteCloser].
// Writes to the returned [io.WriteCloser] are compressed and written to w.
// It is the caller's responsibility to call Close on the WriteCloser when
// finished writing.
// The number of bits to use for literal codes, litWidth, must be in the
// range [2,8] and is typically 8. Input bytes must be less than 1<<litWidth.
//
// It is guaranteed that the underlying type of the returned [io.WriteCloser]
// is a *[Writer].
func NewWriter(w io.Writer, order Order, litWidth int) io.WriteCloser {
	return newWriter(w, order, litWidth)
}

func newWriter(dst io.Writer, order Order, litWidth int) *Writer {
	w := new(Writer)
	w.init(dst, order, litWidth)
	return w
}

func (w *Writer) init(dst io.Writer, order Order, litWidth int) {
	switch order {
	case LSB:
		w.write = (*Writer).writeLSB
	case MSB:
		w.write = (*Writer).writeMSB
	default:
		w.err = errors.New("lzw: unknown order")
		return
	}
	if litWidth < 2 || 8 < litWidth {
		w.err = fmt.Errorf("lzw: litWidth %d out of range", litWidth)
		return
	}
	bw, ok := dst.(writer)
	if !ok && dst != nil {
		bw = bufio.NewWriter(dst)
	}
	w.w = bw
	lw := uint(litWidth)
	w.order = order
	w.width = 1 + lw
	w.litWidth = lw
	w.hi = 1<<lw + 1
	w.overflow = 1 << (lw + 1)
	w.savedCode = invalidCode
}
