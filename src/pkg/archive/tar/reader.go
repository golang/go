// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

// TODO(dsymonds):
//   - pax extensions

import (
	"bytes"
	"io"
	"os"
	"strconv"
)

var (
	HeaderError os.Error = os.ErrorString("invalid tar header")
)

// A Reader provides sequential access to the contents of a tar archive.
// A tar archive consists of a sequence of files.
// The Next method advances to the next file in the archive (including the first),
// and then it can be treated as an io.Reader to access the file's data.
//
// Example:
//	tr := tar.NewReader(r)
//	for {
//		hdr, err := tr.Next()
//		if err != nil {
//			// handle error
//		}
//		if hdr == nil {
//			// end of tar archive
//			break
//		}
//		io.Copy(data, tr)
//	}
type Reader struct {
	r   io.Reader
	err os.Error
	nb  int64 // number of unread bytes for current file entry
	pad int64 // amount of padding (ignored) after current file entry
}

// NewReader creates a new Reader reading from r.
func NewReader(r io.Reader) *Reader { return &Reader{r: r} }

// Next advances to the next entry in the tar archive.
func (tr *Reader) Next() (*Header, os.Error) {
	var hdr *Header
	if tr.err == nil {
		tr.skipUnread()
	}
	if tr.err == nil {
		hdr = tr.readHeader()
	}
	return hdr, tr.err
}

// Parse bytes as a NUL-terminated C-style string.
// If a NUL byte is not found then the whole slice is returned as a string.
func cString(b []byte) string {
	n := 0
	for n < len(b) && b[n] != 0 {
		n++
	}
	return string(b[0:n])
}

func (tr *Reader) octal(b []byte) int64 {
	// Removing leading spaces.
	for len(b) > 0 && b[0] == ' ' {
		b = b[1:]
	}
	// Removing trailing NULs and spaces.
	for len(b) > 0 && (b[len(b)-1] == ' ' || b[len(b)-1] == '\x00') {
		b = b[0 : len(b)-1]
	}
	x, err := strconv.Btoui64(cString(b), 8)
	if err != nil {
		tr.err = err
	}
	return int64(x)
}

type ignoreWriter struct{}

func (ignoreWriter) Write(b []byte) (n int, err os.Error) {
	return len(b), nil
}

// Skip any unread bytes in the existing file entry, as well as any alignment padding.
func (tr *Reader) skipUnread() {
	nr := tr.nb + tr.pad // number of bytes to skip
	tr.nb, tr.pad = 0, 0
	if sr, ok := tr.r.(io.Seeker); ok {
		if _, err := sr.Seek(nr, 1); err == nil {
			return
		}
	}
	_, tr.err = io.Copyn(ignoreWriter{}, tr.r, nr)
}

func (tr *Reader) verifyChecksum(header []byte) bool {
	if tr.err != nil {
		return false
	}

	given := tr.octal(header[148:156])
	unsigned, signed := checksum(header)
	return given == unsigned || given == signed
}

func (tr *Reader) readHeader() *Header {
	header := make([]byte, blockSize)
	if _, tr.err = io.ReadFull(tr.r, header); tr.err != nil {
		return nil
	}

	// Two blocks of zero bytes marks the end of the archive.
	if bytes.Equal(header, zeroBlock[0:blockSize]) {
		if _, tr.err = io.ReadFull(tr.r, header); tr.err != nil {
			return nil
		}
		if bytes.Equal(header, zeroBlock[0:blockSize]) {
			tr.err = os.EOF
		} else {
			tr.err = HeaderError // zero block and then non-zero block
		}
		return nil
	}

	if !tr.verifyChecksum(header) {
		tr.err = HeaderError
		return nil
	}

	// Unpack
	hdr := new(Header)
	s := slicer(header)

	hdr.Name = cString(s.next(100))
	hdr.Mode = tr.octal(s.next(8))
	hdr.Uid = int(tr.octal(s.next(8)))
	hdr.Gid = int(tr.octal(s.next(8)))
	hdr.Size = tr.octal(s.next(12))
	hdr.Mtime = tr.octal(s.next(12))
	s.next(8) // chksum
	hdr.Typeflag = s.next(1)[0]
	hdr.Linkname = cString(s.next(100))

	// The remainder of the header depends on the value of magic.
	// The original (v7) version of tar had no explicit magic field,
	// so its magic bytes, like the rest of the block, are NULs.
	magic := string(s.next(8)) // contains version field as well.
	var format string
	switch magic {
	case "ustar\x0000": // POSIX tar (1003.1-1988)
		if string(header[508:512]) == "tar\x00" {
			format = "star"
		} else {
			format = "posix"
		}
	case "ustar  \x00": // old GNU tar
		format = "gnu"
	}

	switch format {
	case "posix", "gnu", "star":
		hdr.Uname = cString(s.next(32))
		hdr.Gname = cString(s.next(32))
		devmajor := s.next(8)
		devminor := s.next(8)
		if hdr.Typeflag == TypeChar || hdr.Typeflag == TypeBlock {
			hdr.Devmajor = tr.octal(devmajor)
			hdr.Devminor = tr.octal(devminor)
		}
		var prefix string
		switch format {
		case "posix", "gnu":
			prefix = cString(s.next(155))
		case "star":
			prefix = cString(s.next(131))
			hdr.Atime = tr.octal(s.next(12))
			hdr.Ctime = tr.octal(s.next(12))
		}
		if len(prefix) > 0 {
			hdr.Name = prefix + "/" + hdr.Name
		}
	}

	if tr.err != nil {
		tr.err = HeaderError
		return nil
	}

	// Maximum value of hdr.Size is 64 GB (12 octal digits),
	// so there's no risk of int64 overflowing.
	tr.nb = int64(hdr.Size)
	tr.pad = -tr.nb & (blockSize - 1) // blockSize is a power of two

	return hdr
}

// Read reads from the current entry in the tar archive.
// It returns 0, os.EOF when it reaches the end of that entry,
// until Next is called to advance to the next entry.
func (tr *Reader) Read(b []byte) (n int, err os.Error) {
	if tr.nb == 0 {
		// file consumed
		return 0, os.EOF
	}

	if int64(len(b)) > tr.nb {
		b = b[0:tr.nb]
	}
	n, err = tr.r.Read(b)
	tr.nb -= int64(n)

	if err == os.EOF && tr.nb > 0 {
		err = io.ErrUnexpectedEOF
	}
	tr.err = err
	return
}
