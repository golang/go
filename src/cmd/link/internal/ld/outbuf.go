// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bufio"
	"cmd/internal/sys"
	"encoding/binary"
	"os"
)

// OutBuf is a buffered file writer.
//
// It is simlar to the Writer in cmd/internal/bio with a few small differences.
//
// First, it tracks the output architecture and uses it to provide
// endian helpers.
//
// Second, it provides a very cheap offset counter that doesn't require
// any system calls to read the value.
type OutBuf struct {
	arch   *sys.Arch
	off    int64
	w      *bufio.Writer
	f      *os.File
	encbuf [8]byte // temp buffer used by WriteN methods
}

func (out *OutBuf) SeekSet(p int64) {
	if p == out.off {
		return
	}
	out.Flush()
	if _, err := out.f.Seek(p, 0); err != nil {
		Exitf("seeking to %d in %s: %v", p, out.f.Name(), err)
	}
	out.off = p
}

func (out *OutBuf) Offset() int64 {
	return out.off
}

// Write writes the contents of v to the buffer.
//
// As Write is backed by a bufio.Writer, callers do not have
// to explicitly handle the returned error as long as Flush is
// eventually called.
func (out *OutBuf) Write(v []byte) (int, error) {
	n, err := out.w.Write(v)
	out.off += int64(n)
	return n, err
}

func (out *OutBuf) Write8(v uint8) {
	if err := out.w.WriteByte(v); err == nil {
		out.off++
	}
}

// WriteByte is an alias for Write8 to fulfill the io.ByteWriter interface.
func (out *OutBuf) WriteByte(v byte) error {
	out.Write8(v)
	return nil
}

func (out *OutBuf) Write16(v uint16) {
	out.arch.ByteOrder.PutUint16(out.encbuf[:], v)
	out.Write(out.encbuf[:2])
}

func (out *OutBuf) Write32(v uint32) {
	out.arch.ByteOrder.PutUint32(out.encbuf[:], v)
	out.Write(out.encbuf[:4])
}

func (out *OutBuf) Write32b(v uint32) {
	binary.BigEndian.PutUint32(out.encbuf[:], v)
	out.Write(out.encbuf[:4])
}

func (out *OutBuf) Write64(v uint64) {
	out.arch.ByteOrder.PutUint64(out.encbuf[:], v)
	out.Write(out.encbuf[:8])
}

func (out *OutBuf) Write64b(v uint64) {
	binary.BigEndian.PutUint64(out.encbuf[:], v)
	out.Write(out.encbuf[:8])
}

func (out *OutBuf) WriteString(s string) {
	n, _ := out.w.WriteString(s)
	out.off += int64(n)
}

// WriteStringN writes the first n bytes of s.
// If n is larger than len(s) then it is padded with zero bytes.
func (out *OutBuf) WriteStringN(s string, n int) {
	out.WriteStringPad(s, n, zeros[:])
}

// WriteStringPad writes the first n bytes of s.
// If n is larger than len(s) then it is padded with the bytes in pad (repeated as needed).
func (out *OutBuf) WriteStringPad(s string, n int, pad []byte) {
	if len(s) >= n {
		out.WriteString(s[:n])
	} else {
		out.WriteString(s)
		n -= len(s)
		for n > len(pad) {
			out.Write(pad)
			n -= len(pad)

		}
		out.Write(pad[:n])
	}
}

func (out *OutBuf) Flush() {
	if err := out.w.Flush(); err != nil {
		Exitf("flushing %s: %v", out.f.Name(), err)
	}
}
