// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzw

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
)

// A writer is a buffered, flushable writer.
type writer interface {
	WriteByte(byte) error
	Flush() error
}

// An errWriteCloser is an io.WriteCloser that always returns a given error.
type errWriteCloser struct {
	err error
}

func (e *errWriteCloser) Write([]byte) (int, error) {
	return 0, e.err
}

func (e *errWriteCloser) Close() error {
	return e.err
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

// encoder is LZW compressor.
type encoder struct {
	// w is the writer that compressed bytes are written to.
	w writer
	// write, bits, nBits and width are the state for converting a code stream
	// into a byte stream.
	write func(*encoder, uint32) error
	bits  uint32
	nBits uint
	width uint
	// litWidth is the width in bits of literal codes.
	litWidth uint
	// hi is the code implied by the next code emission.
	// overflow is the code at which hi overflows the code width.
	hi, overflow uint32
	// savedCode is the accumulated code at the end of the most recent Write
	// call. It is equal to invalidCode if there was no such call.
	savedCode uint32
	// err is the first error encountered during writing. Closing the encoder
	// will make any future Write calls return os.EINVAL.
	err error
	// table is the hash table from 20-bit keys to 12-bit values. Each table
	// entry contains key<<12|val and collisions resolve by linear probing.
	// The keys consist of a 12-bit code prefix and an 8-bit byte suffix.
	// The values are a 12-bit code.
	table [tableSize]uint32
}

// writeLSB writes the code c for "Least Significant Bits first" data.
func (e *encoder) writeLSB(c uint32) error {
	e.bits |= c << e.nBits
	e.nBits += e.width
	for e.nBits >= 8 {
		if err := e.w.WriteByte(uint8(e.bits)); err != nil {
			return err
		}
		e.bits >>= 8
		e.nBits -= 8
	}
	return nil
}

// writeMSB writes the code c for "Most Significant Bits first" data.
func (e *encoder) writeMSB(c uint32) error {
	e.bits |= c << (32 - e.width - e.nBits)
	e.nBits += e.width
	for e.nBits >= 8 {
		if err := e.w.WriteByte(uint8(e.bits >> 24)); err != nil {
			return err
		}
		e.bits <<= 8
		e.nBits -= 8
	}
	return nil
}

// errOutOfCodes is an internal error that means that the encoder has run out
// of unused codes and a clear code needs to be sent next.
var errOutOfCodes = errors.New("lzw: out of codes")

// incHi increments e.hi and checks for both overflow and running out of
// unused codes. In the latter case, incHi sends a clear code, resets the
// encoder state and returns errOutOfCodes.
func (e *encoder) incHi() error {
	e.hi++
	if e.hi == e.overflow {
		e.width++
		e.overflow <<= 1
	}
	if e.hi == maxCode {
		clear := uint32(1) << e.litWidth
		if err := e.write(e, clear); err != nil {
			return err
		}
		e.width = uint(e.litWidth) + 1
		e.hi = clear + 1
		e.overflow = clear << 1
		for i := range e.table {
			e.table[i] = invalidEntry
		}
		return errOutOfCodes
	}
	return nil
}

// Write writes a compressed representation of p to e's underlying writer.
func (e *encoder) Write(p []byte) (int, error) {
	if e.err != nil {
		return 0, e.err
	}
	if len(p) == 0 {
		return 0, nil
	}
	litMask := uint32(1<<e.litWidth - 1)
	code := e.savedCode
	if code == invalidCode {
		// The first code sent is always a literal code.
		code, p = uint32(p[0])&litMask, p[1:]
	}
loop:
	for _, x := range p {
		literal := uint32(x) & litMask
		key := code<<8 | literal
		// If there is a hash table hit for this key then we continue the loop
		// and do not emit a code yet.
		hash := (key>>12 ^ key) & tableMask
		for h, t := hash, e.table[hash]; t != invalidEntry; {
			if key == t>>12 {
				code = t & maxCode
				continue loop
			}
			h = (h + 1) & tableMask
			t = e.table[h]
		}
		// Otherwise, write the current code, and literal becomes the start of
		// the next emitted code.
		if e.err = e.write(e, code); e.err != nil {
			return 0, e.err
		}
		code = literal
		// Increment e.hi, the next implied code. If we run out of codes, reset
		// the encoder state (including clearing the hash table) and continue.
		if err := e.incHi(); err != nil {
			if err == errOutOfCodes {
				continue
			}
			e.err = err
			return 0, e.err
		}
		// Otherwise, insert key -> e.hi into the map that e.table represents.
		for {
			if e.table[hash] == invalidEntry {
				e.table[hash] = (key << 12) | e.hi
				break
			}
			hash = (hash + 1) & tableMask
		}
	}
	e.savedCode = code
	return len(p), nil
}

// Close closes the encoder, flushing any pending output. It does not close or
// flush e's underlying writer.
func (e *encoder) Close() error {
	if e.err != nil {
		if e.err == os.EINVAL {
			return nil
		}
		return e.err
	}
	// Make any future calls to Write return os.EINVAL.
	e.err = os.EINVAL
	// Write the savedCode if valid.
	if e.savedCode != invalidCode {
		if err := e.write(e, e.savedCode); err != nil {
			return err
		}
		if err := e.incHi(); err != nil && err != errOutOfCodes {
			return err
		}
	}
	// Write the eof code.
	eof := uint32(1)<<e.litWidth + 1
	if err := e.write(e, eof); err != nil {
		return err
	}
	// Write the final bits.
	if e.nBits > 0 {
		if e.write == (*encoder).writeMSB {
			e.bits >>= 24
		}
		if err := e.w.WriteByte(uint8(e.bits)); err != nil {
			return err
		}
	}
	return e.w.Flush()
}

// NewWriter creates a new io.WriteCloser that satisfies writes by compressing
// the data and writing it to w.
// It is the caller's responsibility to call Close on the WriteCloser when
// finished writing.
// The number of bits to use for literal codes, litWidth, must be in the
// range [2,8] and is typically 8.
func NewWriter(w io.Writer, order Order, litWidth int) io.WriteCloser {
	var write func(*encoder, uint32) error
	switch order {
	case LSB:
		write = (*encoder).writeLSB
	case MSB:
		write = (*encoder).writeMSB
	default:
		return &errWriteCloser{errors.New("lzw: unknown order")}
	}
	if litWidth < 2 || 8 < litWidth {
		return &errWriteCloser{fmt.Errorf("lzw: litWidth %d out of range", litWidth)}
	}
	bw, ok := w.(writer)
	if !ok {
		bw = bufio.NewWriter(w)
	}
	lw := uint(litWidth)
	return &encoder{
		w:         bw,
		write:     write,
		width:     1 + lw,
		litWidth:  lw,
		hi:        1<<lw + 1,
		overflow:  1 << (lw + 1),
		savedCode: invalidCode,
	}
}
