// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package filebuf implements io.SeekReader for os files.
// This is useful only for very large files with lots of
// seeking. (otherwise use ioutil.ReadFile or bufio)
package filebuf

import (
	"fmt"
	"io"
	"os"
)

// Buf is the implemented interface
type Buf interface {
	io.ReadCloser
	io.Seeker
	Size() int64
	Stats() Stat
}

// Buflen is the size of the internal buffer.
// The code is designed to never need to reread unnecessarily
const Buflen = 1 << 20

// fbuf is a buffered file with seeking.
// fixed is an internal buffer. buf is the slice fixed[:fixedLen]. bufloc is the file
// location of the beginning of fixed (and buf). The seek pointer is at bufloc+bufpos,
// so the file's contents there start with buf[bufpos:]
type fbuf struct {
	Name     string
	fd       *os.File
	size     int64        // file size
	bufloc   int64        // file loc of beginning of fixed
	bufpos   int32        // seekptr is at bufloc+bufpos. bufpos  <= Buflen, fixedLen
	fixed    [Buflen]byte // backing store for buf
	fixedlen int          // how much of fixed is valid file contents
	buf      []byte       // buf is fixed[0:fixedlen]
	// statistics
	seeks int   // number of calls to fd.Seek
	reads int   // number of calls to fd.Read
	bytes int64 // number of bytes read by fd.Read
}

// Stat returns the number of underlying seeks and reads, and bytes read
type Stat struct {
	Seeks int
	Reads int
	Bytes int64
}

// Stats returns the stats so far
func (fb *fbuf) Stats() Stat {
	return Stat{fb.seeks, fb.reads, fb.bytes}
}

// Size returns the file size
func (fb *fbuf) Size() int64 {
	return fb.size
}

// New returns an initialized *fbuf or an error
func New(fname string) (Buf, error) {
	fd, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	fi, err := fd.Stat()
	if err != nil || fi.Mode().IsDir() {
		return nil, fmt.Errorf("not readable: %s", fname)
	}
	return &fbuf{Name: fname, fd: fd, size: fi.Size()}, nil
}

// Read implements io.Reader. It may return a positive
// number of bytes read with io.EOF
func (fb *fbuf) Read(p []byte) (int, error) {
	// If there are enough valid bytes remaining in buf, just use them
	if len(fb.buf[fb.bufpos:]) >= len(p) {
		copy(p, fb.buf[fb.bufpos:])
		fb.bufpos += int32(len(p))
		return len(p), nil
	}
	done := 0 // done counts how many bytes have been transferred
	// If there are any valid bytes left in buf, use them first
	if len(fb.buf[fb.bufpos:]) > 0 {
		m := copy(p, fb.buf[fb.bufpos:])
		done = m
		fb.bufpos += int32(done) // at end of the valid bytes in buf
	}
	// used up buffered data. logical seek pointer is at bufloc+bufpos.
	// loop until p has been filled up or EOF
	for done < len(p) {
		loc, err := fb.fd.Seek(0, io.SeekCurrent) // make sure of the os's file location
		if loc != fb.bufloc+int64(fb.bufpos) {
			panic(fmt.Sprintf("%v loc=%d bufloc=%d bufpos=%d", err, loc,
				fb.bufloc, fb.bufpos))
		}
		fb.seeks++ // did a file system seek
		if loc >= fb.size {
			// at EOF
			fb.bufpos = int32(len(fb.buf))
			fb.bufloc = loc - int64(fb.fixedlen)
			return done, io.EOF
		}
		n, err := fb.fd.Read(fb.fixed[:])
		if n != 0 {
			fb.fixedlen = n
		}
		fb.reads++ // did a file system read
		m := copy(p[done:], fb.fixed[:n])
		done += m
		if err != nil {
			if err == io.EOF {
				fb.bufpos = int32(len(fb.buf))
				fb.bufloc = loc - int64(fb.fixedlen)
				return done, io.EOF
			}
			return 0, err
		}
		fb.bytes += int64(n)
		fb.bufpos = int32(m) // used m byes of the buffer
		fb.bufloc = loc
		fb.buf = fb.fixed[:n]
	}
	return len(p), nil
}

// Seek implements io.Seeker. (<unchanged>, io.EOF) is returned for seeks off the end.
func (fb *fbuf) Seek(offset int64, whence int) (int64, error) {
	seekpos := offset
	switch whence {
	case io.SeekCurrent:
		seekpos += fb.bufloc + int64(fb.bufpos)
	case io.SeekEnd:
		seekpos += fb.size
	}
	if seekpos < 0 || seekpos > fb.size {
		return fb.bufloc + int64(fb.bufpos), io.EOF
	}
	// if seekpos is inside fixed, just adjust buf and bufpos
	if seekpos >= fb.bufloc && seekpos <= int64(fb.fixedlen)+fb.bufloc {
		fb.bufpos = int32(seekpos - fb.bufloc)
		return seekpos, nil
	}
	// need to refresh the internal buffer. Seek does no reading, mark buf
	// as empty, set bufpos and bufloc.
	fb.buf, fb.bufpos, fb.bufloc = nil, 0, seekpos
	n, err := fb.fd.Seek(seekpos, io.SeekStart)
	fb.seeks++
	if n != seekpos || err != nil {
		return -1, fmt.Errorf("seek failed (%d!= %d) %v", n, seekpos, err)
	}
	return seekpos, nil
}

// Close closes the underlying file
func (fb *fbuf) Close() error {
	if fb.fd != nil {
		return fb.fd.Close()
	}
	return nil
}
