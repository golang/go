// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin linux

package core

import (
	"errors"
	"fmt"
	"io"
	"os"
	"syscall"
)

var errMmapClosed = errors.New("mmap: closed")

// mmapFile wraps a memory-mapped file.
type mmapFile struct {
	data     []byte
	pos      uint64
	writable bool
}

// mmapOpen opens the named file for reading.
// If writable is true, the file is also open for writing.
func mmapOpen(filename string, writable bool) (*mmapFile, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}

	size := st.Size()
	if size == 0 {
		return &mmapFile{data: []byte{}}, nil
	}
	if size < 0 {
		return nil, fmt.Errorf("mmap: file %q has negative size: %d", filename, size)
	}
	if size != int64(int(size)) {
		return nil, fmt.Errorf("mmap: file %q is too large", filename)
	}

	prot := syscall.PROT_READ
	if writable {
		prot |= syscall.PROT_WRITE
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), prot, syscall.MAP_SHARED)
	if err != nil {
		return nil, err
	}
	return &mmapFile{data: data, writable: writable}, nil
}

// Size returns the size of the mapped file.
func (f *mmapFile) Size() uint64 {
	return uint64(len(f.data))
}

// Pos returns the current file pointer.
func (f *mmapFile) Pos() uint64 {
	return f.pos
}

// SeekTo sets the current file pointer relative to the start of the file.
func (f *mmapFile) SeekTo(offset uint64) {
	f.pos = offset
}

// Read implements io.Reader.
func (f *mmapFile) Read(p []byte) (int, error) {
	if f.data == nil {
		return 0, errMmapClosed
	}
	if f.pos >= f.Size() {
		return 0, io.EOF
	}
	n := copy(p, f.data[f.pos:])
	f.pos += uint64(n)
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

// ReadByte implements io.ByteReader.
func (f *mmapFile) ReadByte() (byte, error) {
	if f.data == nil {
		return 0, errMmapClosed
	}
	if f.pos >= f.Size() {
		return 0, io.EOF
	}
	b := f.data[f.pos]
	f.pos++
	return b, nil
}

// ReadSlice returns a slice of size n that points directly at the
// underlying mapped file. There is no copying. Fails if it cannot
// read at least n bytes.
func (f *mmapFile) ReadSlice(n uint64) ([]byte, error) {
	if f.data == nil {
		return nil, errMmapClosed
	}
	if f.pos+n >= f.Size() {
		return nil, io.EOF
	}
	first := f.pos
	f.pos += n
	return f.data[first:f.pos:f.pos], nil
}

// ReadSliceAt is like ReadSlice, but reads from a specific offset.
// The file pointer is not used or advanced.
func (f *mmapFile) ReadSliceAt(offset, n uint64) ([]byte, error) {
	if f.data == nil {
		return nil, errMmapClosed
	}
	if f.Size() < offset {
		return nil, fmt.Errorf("mmap: out-of-bounds ReadSliceAt offset %d, size is %d", offset, f.Size())
	}
	if offset+n >= f.Size() {
		return nil, io.EOF
	}
	end := offset + n
	return f.data[offset:end:end], nil
}

// Close closes the file.
func (f *mmapFile) Close() error {
	if f.data == nil {
		return nil
	}
	err := syscall.Munmap(f.data)
	f.data = nil
	f.pos = 0
	return err
}
