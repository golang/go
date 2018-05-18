// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import (
	"io"
)

// Install Native Client stdin, stdout, stderr.
func init() {
	newFD(&naclFile{naclFD: 0})
	newFD(&naclFile{naclFD: 1})
	newFD(&naclFile{naclFD: 2})
}

// naclFile is the fileImpl implementation for a Native Client file descriptor.
type naclFile struct {
	defaultFileImpl
	naclFD int
}

func (f *naclFile) stat(st *Stat_t) error {
	return naclFstat(f.naclFD, st)
}

func (f *naclFile) read(b []byte) (int, error) {
	n, err := naclRead(f.naclFD, b)
	if err != nil {
		n = 0
	}
	return n, err
}

// implemented in package runtime, to add time header on playground
func naclWrite(fd int, b []byte) int

func (f *naclFile) write(b []byte) (int, error) {
	n := naclWrite(f.naclFD, b)
	if n < 0 {
		return 0, Errno(-n)
	}
	return n, nil
}

func (f *naclFile) seek(off int64, whence int) (int64, error) {
	old := off
	err := naclSeek(f.naclFD, &off, whence)
	if err != nil {
		return old, err
	}
	return off, nil
}

func (f *naclFile) prw(b []byte, offset int64, rw func([]byte) (int, error)) (int, error) {
	// NaCl has no pread; simulate with seek and hope for no races.
	old, err := f.seek(0, io.SeekCurrent)
	if err != nil {
		return 0, err
	}
	if _, err := f.seek(offset, io.SeekStart); err != nil {
		return 0, err
	}
	n, err := rw(b)
	f.seek(old, io.SeekStart)
	return n, err
}

func (f *naclFile) pread(b []byte, offset int64) (int, error) {
	return f.prw(b, offset, f.read)
}

func (f *naclFile) pwrite(b []byte, offset int64) (int, error) {
	return f.prw(b, offset, f.write)
}

func (f *naclFile) close() error {
	err := naclClose(f.naclFD)
	f.naclFD = -1
	return err
}
