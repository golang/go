// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

// TODO(dsymonds):
//   - pax extensions

import (
	"bytes"
	"errors"
	"io"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
	"time"
)

var (
	ErrHeader = errors.New("archive/tar: invalid tar header")
)

const maxNanoSecondIntSize = 9

// A Reader provides sequential access to the contents of a tar archive.
// A tar archive consists of a sequence of files.
// The Next method advances to the next file in the archive (including the first),
// and then it can be treated as an io.Reader to access the file's data.
type Reader struct {
	r   io.Reader
	err error
	nb  int64 // number of unread bytes for current file entry
	pad int64 // amount of padding (ignored) after current file entry
}

// NewReader creates a new Reader reading from r.
func NewReader(r io.Reader) *Reader { return &Reader{r: r} }

// Next advances to the next entry in the tar archive.
func (tr *Reader) Next() (*Header, error) {
	var hdr *Header
	if tr.err == nil {
		tr.skipUnread()
	}
	if tr.err != nil {
		return hdr, tr.err
	}
	hdr = tr.readHeader()
	if hdr == nil {
		return hdr, tr.err
	}
	// Check for PAX/GNU header.
	switch hdr.Typeflag {
	case TypeXHeader:
		//  PAX extended header
		headers, err := parsePAX(tr)
		if err != nil {
			return nil, err
		}
		// We actually read the whole file,
		// but this skips alignment padding
		tr.skipUnread()
		hdr = tr.readHeader()
		mergePAX(hdr, headers)
		return hdr, nil
	case TypeGNULongName:
		// We have a GNU long name header. Its contents are the real file name.
		realname, err := ioutil.ReadAll(tr)
		if err != nil {
			return nil, err
		}
		hdr, err := tr.Next()
		hdr.Name = cString(realname)
		return hdr, err
	case TypeGNULongLink:
		// We have a GNU long link header.
		realname, err := ioutil.ReadAll(tr)
		if err != nil {
			return nil, err
		}
		hdr, err := tr.Next()
		hdr.Linkname = cString(realname)
		return hdr, err
	}
	return hdr, tr.err
}

// mergePAX merges well known headers according to PAX standard.
// In general headers with the same name as those found
// in the header struct overwrite those found in the header
// struct with higher precision or longer values. Esp. useful
// for name and linkname fields.
func mergePAX(hdr *Header, headers map[string]string) error {
	for k, v := range headers {
		switch k {
		case "path":
			hdr.Name = v
		case "linkpath":
			hdr.Linkname = v
		case "gname":
			hdr.Gname = v
		case "uname":
			hdr.Uname = v
		case "uid":
			uid, err := strconv.ParseInt(v, 10, 0)
			if err != nil {
				return err
			}
			hdr.Uid = int(uid)
		case "gid":
			gid, err := strconv.ParseInt(v, 10, 0)
			if err != nil {
				return err
			}
			hdr.Gid = int(gid)
		case "atime":
			t, err := parsePAXTime(v)
			if err != nil {
				return err
			}
			hdr.AccessTime = t
		case "mtime":
			t, err := parsePAXTime(v)
			if err != nil {
				return err
			}
			hdr.ModTime = t
		case "ctime":
			t, err := parsePAXTime(v)
			if err != nil {
				return err
			}
			hdr.ChangeTime = t
		case "size":
			size, err := strconv.ParseInt(v, 10, 0)
			if err != nil {
				return err
			}
			hdr.Size = int64(size)
		}

	}
	return nil
}

// parsePAXTime takes a string of the form %d.%d as described in
// the PAX specification.
func parsePAXTime(t string) (time.Time, error) {
	buf := []byte(t)
	pos := bytes.IndexByte(buf, '.')
	var seconds, nanoseconds int64
	var err error
	if pos == -1 {
		seconds, err = strconv.ParseInt(t, 10, 0)
		if err != nil {
			return time.Time{}, err
		}
	} else {
		seconds, err = strconv.ParseInt(string(buf[:pos]), 10, 0)
		if err != nil {
			return time.Time{}, err
		}
		nano_buf := string(buf[pos+1:])
		// Pad as needed before converting to a decimal.
		// For example .030 -> .030000000 -> 30000000 nanoseconds
		if len(nano_buf) < maxNanoSecondIntSize {
			// Right pad
			nano_buf += strings.Repeat("0", maxNanoSecondIntSize-len(nano_buf))
		} else if len(nano_buf) > maxNanoSecondIntSize {
			// Right truncate
			nano_buf = nano_buf[:maxNanoSecondIntSize]
		}
		nanoseconds, err = strconv.ParseInt(string(nano_buf), 10, 0)
		if err != nil {
			return time.Time{}, err
		}
	}
	ts := time.Unix(seconds, nanoseconds)
	return ts, nil
}

// parsePAX parses PAX headers.
// If an extended header (type 'x') is invalid, ErrHeader is returned
func parsePAX(r io.Reader) (map[string]string, error) {
	buf, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	headers := make(map[string]string)
	// Each record is constructed as
	//     "%d %s=%s\n", length, keyword, value
	for len(buf) > 0 {
		// or the header was empty to start with.
		var sp int
		// The size field ends at the first space.
		sp = bytes.IndexByte(buf, ' ')
		if sp == -1 {
			return nil, ErrHeader
		}
		// Parse the first token as a decimal integer.
		n, err := strconv.ParseInt(string(buf[:sp]), 10, 0)
		if err != nil {
			return nil, ErrHeader
		}
		// Extract everything between the decimal and the n -1 on the
		// beginning to to eat the ' ', -1 on the end to skip the newline.
		var record []byte
		record, buf = buf[sp+1:n-1], buf[n:]
		// The first equals is guaranteed to mark the end of the key.
		// Everything else is value.
		eq := bytes.IndexByte(record, '=')
		if eq == -1 {
			return nil, ErrHeader
		}
		key, value := record[:eq], record[eq+1:]
		headers[string(key)] = string(value)
	}
	return headers, nil
}

// cString parses bytes as a NUL-terminated C-style string.
// If a NUL byte is not found then the whole slice is returned as a string.
func cString(b []byte) string {
	n := 0
	for n < len(b) && b[n] != 0 {
		n++
	}
	return string(b[0:n])
}

func (tr *Reader) octal(b []byte) int64 {
	// Check for binary format first.
	if len(b) > 0 && b[0]&0x80 != 0 {
		var x int64
		for i, c := range b {
			if i == 0 {
				c &= 0x7f // ignore signal bit in first byte
			}
			x = x<<8 | int64(c)
		}
		return x
	}

	// Removing leading spaces.
	for len(b) > 0 && b[0] == ' ' {
		b = b[1:]
	}
	// Removing trailing NULs and spaces.
	for len(b) > 0 && (b[len(b)-1] == ' ' || b[len(b)-1] == '\x00') {
		b = b[0 : len(b)-1]
	}
	x, err := strconv.ParseUint(cString(b), 8, 64)
	if err != nil {
		tr.err = err
	}
	return int64(x)
}

// skipUnread skips any unread bytes in the existing file entry, as well as any alignment padding.
func (tr *Reader) skipUnread() {
	nr := tr.nb + tr.pad // number of bytes to skip
	tr.nb, tr.pad = 0, 0
	if sr, ok := tr.r.(io.Seeker); ok {
		if _, err := sr.Seek(nr, os.SEEK_CUR); err == nil {
			return
		}
	}
	_, tr.err = io.CopyN(ioutil.Discard, tr.r, nr)
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
			tr.err = io.EOF
		} else {
			tr.err = ErrHeader // zero block and then non-zero block
		}
		return nil
	}

	if !tr.verifyChecksum(header) {
		tr.err = ErrHeader
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
	hdr.ModTime = time.Unix(tr.octal(s.next(12)), 0)
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
			hdr.AccessTime = time.Unix(tr.octal(s.next(12)), 0)
			hdr.ChangeTime = time.Unix(tr.octal(s.next(12)), 0)
		}
		if len(prefix) > 0 {
			hdr.Name = prefix + "/" + hdr.Name
		}
	}

	if tr.err != nil {
		tr.err = ErrHeader
		return nil
	}

	// Maximum value of hdr.Size is 64 GB (12 octal digits),
	// so there's no risk of int64 overflowing.
	tr.nb = int64(hdr.Size)
	tr.pad = -tr.nb & (blockSize - 1) // blockSize is a power of two

	return hdr
}

// Read reads from the current entry in the tar archive.
// It returns 0, io.EOF when it reaches the end of that entry,
// until Next is called to advance to the next entry.
func (tr *Reader) Read(b []byte) (n int, err error) {
	if tr.nb == 0 {
		// file consumed
		return 0, io.EOF
	}

	if int64(len(b)) > tr.nb {
		b = b[0:tr.nb]
	}
	n, err = tr.r.Read(b)
	tr.nb -= int64(n)

	if err == io.EOF && tr.nb > 0 {
		err = io.ErrUnexpectedEOF
	}
	tr.err = err
	return
}
