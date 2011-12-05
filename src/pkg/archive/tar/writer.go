// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

// TODO(dsymonds):
// - catch more errors (no first header, write after close, etc.)

import (
	"errors"
	"io"
	"strconv"
)

var (
	ErrWriteTooLong    = errors.New("write too long")
	ErrFieldTooLong    = errors.New("header field too long")
	ErrWriteAfterClose = errors.New("write after close")
)

// A Writer provides sequential writing of a tar archive in POSIX.1 format.
// A tar archive consists of a sequence of files.
// Call WriteHeader to begin a new file, and then call Write to supply that file's data,
// writing at most hdr.Size bytes in total.
//
// Example:
//	tw := tar.NewWriter(w)
//	hdr := new(Header)
//	hdr.Size = length of data in bytes
//	// populate other hdr fields as desired
//	if err := tw.WriteHeader(hdr); err != nil {
//		// handle error
//	}
//	io.Copy(tw, data)
//	tw.Close()
type Writer struct {
	w          io.Writer
	err        error
	nb         int64 // number of unwritten bytes for current file entry
	pad        int64 // amount of padding to write after current file entry
	closed     bool
	usedBinary bool // whether the binary numeric field extension was used
}

// NewWriter creates a new Writer writing to w.
func NewWriter(w io.Writer) *Writer { return &Writer{w: w} }

// Flush finishes writing the current file (optional).
func (tw *Writer) Flush() error {
	n := tw.nb + tw.pad
	for n > 0 && tw.err == nil {
		nr := n
		if nr > blockSize {
			nr = blockSize
		}
		var nw int
		nw, tw.err = tw.w.Write(zeroBlock[0:nr])
		n -= int64(nw)
	}
	tw.nb = 0
	tw.pad = 0
	return tw.err
}

// Write s into b, terminating it with a NUL if there is room.
func (tw *Writer) cString(b []byte, s string) {
	if len(s) > len(b) {
		if tw.err == nil {
			tw.err = ErrFieldTooLong
		}
		return
	}
	copy(b, s)
	if len(s) < len(b) {
		b[len(s)] = 0
	}
}

// Encode x as an octal ASCII string and write it into b with leading zeros.
func (tw *Writer) octal(b []byte, x int64) {
	s := strconv.FormatInt(x, 8)
	// leading zeros, but leave room for a NUL.
	for len(s)+1 < len(b) {
		s = "0" + s
	}
	tw.cString(b, s)
}

// Write x into b, either as octal or as binary (GNUtar/star extension).
func (tw *Writer) numeric(b []byte, x int64) {
	// Try octal first.
	s := strconv.FormatInt(x, 8)
	if len(s) < len(b) {
		tw.octal(b, x)
		return
	}
	// Too big: use binary (big-endian).
	tw.usedBinary = true
	for i := len(b) - 1; x > 0 && i >= 0; i-- {
		b[i] = byte(x)
		x >>= 8
	}
	b[0] |= 0x80 // highest bit indicates binary format
}

// WriteHeader writes hdr and prepares to accept the file's contents.
// WriteHeader calls Flush if it is not the first header.
// Calling after a Close will return ErrWriteAfterClose.
func (tw *Writer) WriteHeader(hdr *Header) error {
	if tw.closed {
		return ErrWriteAfterClose
	}
	if tw.err == nil {
		tw.Flush()
	}
	if tw.err != nil {
		return tw.err
	}

	tw.nb = int64(hdr.Size)
	tw.pad = -tw.nb & (blockSize - 1) // blockSize is a power of two

	header := make([]byte, blockSize)
	s := slicer(header)

	// TODO(dsymonds): handle names longer than 100 chars
	copy(s.next(100), []byte(hdr.Name))

	tw.octal(s.next(8), hdr.Mode)              // 100:108
	tw.numeric(s.next(8), int64(hdr.Uid))      // 108:116
	tw.numeric(s.next(8), int64(hdr.Gid))      // 116:124
	tw.numeric(s.next(12), hdr.Size)           // 124:136
	tw.numeric(s.next(12), hdr.ModTime.Unix()) // 136:148
	s.next(8)                                  // chksum (148:156)
	s.next(1)[0] = hdr.Typeflag                // 156:157
	tw.cString(s.next(100), hdr.Linkname)      // linkname (157:257)
	copy(s.next(8), []byte("ustar\x0000"))     // 257:265
	tw.cString(s.next(32), hdr.Uname)          // 265:297
	tw.cString(s.next(32), hdr.Gname)          // 297:329
	tw.numeric(s.next(8), hdr.Devmajor)        // 329:337
	tw.numeric(s.next(8), hdr.Devminor)        // 337:345

	// Use the GNU magic instead of POSIX magic if we used any GNU extensions.
	if tw.usedBinary {
		copy(header[257:265], []byte("ustar  \x00"))
	}

	// The chksum field is terminated by a NUL and a space.
	// This is different from the other octal fields.
	chksum, _ := checksum(header)
	tw.octal(header[148:155], chksum)
	header[155] = ' '

	if tw.err != nil {
		// problem with header; probably integer too big for a field.
		return tw.err
	}

	_, tw.err = tw.w.Write(header)

	return tw.err
}

// Write writes to the current entry in the tar archive.
// Write returns the error ErrWriteTooLong if more than
// hdr.Size bytes are written after WriteHeader.
func (tw *Writer) Write(b []byte) (n int, err error) {
	if tw.closed {
		err = ErrWriteTooLong
		return
	}
	overwrite := false
	if int64(len(b)) > tw.nb {
		b = b[0:tw.nb]
		overwrite = true
	}
	n, err = tw.w.Write(b)
	tw.nb -= int64(n)
	if err == nil && overwrite {
		err = ErrWriteTooLong
		return
	}
	tw.err = err
	return
}

// Close closes the tar archive, flushing any unwritten
// data to the underlying writer.
func (tw *Writer) Close() error {
	if tw.err != nil || tw.closed {
		return tw.err
	}
	tw.Flush()
	tw.closed = true

	// trailer: two zero blocks
	for i := 0; i < 2; i++ {
		_, tw.err = tw.w.Write(zeroBlock)
		if tw.err != nil {
			break
		}
	}
	return tw.err
}
