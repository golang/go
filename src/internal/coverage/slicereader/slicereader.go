// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slicereader

import (
	"fmt"
	"internal/binarylite"
	"io"
	"unsafe"
)

// This file contains the helper "SliceReader", a utility for
// reading values from a byte slice that may or may not be backed
// by a read-only mmap'd region.

type Reader struct {
	b        []byte
	readonly bool
	off      int64
}

func NewReader(b []byte, readonly bool) *Reader {
	r := Reader{
		b:        b,
		readonly: readonly,
	}
	return &r
}

func (r *Reader) Read(b []byte) (int, error) {
	amt := len(b)
	toread := r.b[r.off:]
	if len(toread) < amt {
		amt = len(toread)
	}
	copy(b, toread)
	r.off += int64(amt)
	return amt, nil
}

func (r *Reader) Seek(offset int64, whence int) (ret int64, err error) {
	switch whence {
	case io.SeekStart:
		if offset < 0 || offset > int64(len(r.b)) {
			return 0, fmt.Errorf("invalid seek: new offset %d (out of range [0 %d]", offset, len(r.b))
		}
		r.off = offset
		return offset, nil
	case io.SeekCurrent:
		newoff := r.off + offset
		if newoff < 0 || newoff > int64(len(r.b)) {
			return 0, fmt.Errorf("invalid seek: new offset %d (out of range [0 %d]", newoff, len(r.b))
		}
		r.off = newoff
		return r.off, nil
	case io.SeekEnd:
		newoff := int64(len(r.b)) + offset
		if newoff < 0 || newoff > int64(len(r.b)) {
			return 0, fmt.Errorf("invalid seek: new offset %d (out of range [0 %d]", newoff, len(r.b))
		}
		r.off = newoff
		return r.off, nil
	}
	// other modes are not supported
	return 0, fmt.Errorf("unsupported seek mode %d", whence)
}

func (r *Reader) Offset() int64 {
	return r.off
}

func (r *Reader) ReadUint8() uint8 {
	rv := uint8(r.b[int(r.off)])
	r.off += 1
	return rv
}

func (r *Reader) ReadUint32() uint32 {
	end := int(r.off) + 4
	rv := binarylite.LittleEndian.Uint32(r.b[int(r.off):end:end])
	r.off += 4
	return rv
}

func (r *Reader) ReadUint64() uint64 {
	end := int(r.off) + 8
	rv := binarylite.LittleEndian.Uint64(r.b[int(r.off):end:end])
	r.off += 8
	return rv
}

func (r *Reader) ReadULEB128() (value uint64) {
	var shift uint

	for {
		b := r.b[r.off]
		r.off++
		value |= (uint64(b&0x7F) << shift)
		if b&0x80 == 0 {
			break
		}
		shift += 7
	}
	return
}

func (r *Reader) ReadString(len int64) string {
	b := r.b[r.off : r.off+len]
	r.off += len
	if r.readonly {
		return toString(b) // backed by RO memory, ok to make unsafe string
	}
	return string(b)
}

func toString(b []byte) string {
	if len(b) == 0 {
		return ""
	}
	return unsafe.String(&b[0], len(b))
}
