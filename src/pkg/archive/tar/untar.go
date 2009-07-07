// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The tar package implements access to tar archives.
// It aims to cover most of the variations, including those produced
// by GNU and BSD tars.
//
// References:
//   http://www.freebsd.org/cgi/man.cgi?query=tar&sektion=5
//   http://www.gnu.org/software/tar/manual/html_node/Standard.html
package tar

// TODO(dsymonds):
//   - pax extensions

import (
	"bufio";
	"bytes";
	"io";
	"os";
	"strconv";
)

var (
	HeaderError os.Error = os.ErrorString("invalid tar header");
)

// A tar archive consists of a sequence of files.
// A Reader provides sequential access to the contents of a tar archive.
// The Next method advances to the next file in the archive (including the first),
// and then it can be treated as an io.Reader to access the file's data.
//
// Example:
// 	tr := NewTarReader(r);
// 	for {
//		hdr, err := tr.Next();
//		if err != nil {
//			// handle error
//		}
//		if hdr == nil {
//			// end of tar archive
//			break
//		}
//		io.Copy(tr, somewhere);
// 	}
type Reader struct {
	r io.Reader;
	err os.Error;
	nb int64;	// number of unread bytes for current file entry
	pad int64;	// amount of padding (ignored) after current file entry
}

// A Header represents a single header in a tar archive.
// Only some fields may be populated.
type Header struct {
	Name string;
	Mode int64;
	Uid int64;
	Gid int64;
	Size int64;
	Mtime int64;
	Typeflag byte;
	Linkname string;
	Uname string;
	Gname string;
	Devmajor int64;
	Devminor int64;
	Atime int64;
	Ctime int64;
}

func (tr *Reader) skipUnread()
func (tr *Reader) readHeader() *Header

// NewReader creates a new Reader reading the given io.Reader.
func NewReader(r io.Reader) *Reader {
	return &Reader{ r: r }
}

// Next advances to the next entry in the tar archive.
func (tr *Reader) Next() (*Header, os.Error) {
	var hdr *Header;
	if tr.err == nil {
		tr.skipUnread();
	}
	if tr.err == nil {
		hdr = tr.readHeader();
	}
	return hdr, tr.err
}

const (
	blockSize = 512;

	// Types
	TypeReg = '0';
	TypeRegA = '\x00';
	TypeLink = '1';
	TypeSymlink = '2';
	TypeChar = '3';
	TypeBlock = '4';
	TypeDir = '5';
	TypeFifo = '6';
	TypeCont = '7';
	TypeXHeader = 'x';
	TypeXGlobalHeader = 'g';
)

var zeroBlock = make([]byte, blockSize);

// Parse bytes as a NUL-terminated C-style string.
// If a NUL byte is not found then the whole slice is returned as a string.
func cString(b []byte) string {
	n := 0;
	for n < len(b) && b[n] != 0 {
		n++;
	}
	return string(b[0:n])
}

func (tr *Reader) octal(b []byte) int64 {
	if len(b) > 0 && b[len(b)-1] == ' ' {
		b = b[0:len(b)-1];
	}
	x, err := strconv.Btoui64(cString(b), 8);
	if err != nil {
		tr.err = err;
	}
	return int64(x)
}

type ignoreWriter struct {}
func (ignoreWriter) Write(b []byte) (n int, err os.Error) {
	return len(b), nil
}

type seeker interface {
	Seek(offset int64, whence int) (ret int64, err os.Error);
}

// Skip any unread bytes in the existing file entry, as well as any alignment padding.
func (tr *Reader) skipUnread() {
	nr := tr.nb + tr.pad;	// number of bytes to skip

	var n int64;
	if sr, ok := tr.r.(seeker); ok {
		n, tr.err = sr.Seek(nr, 1);
	} else {
		n, tr.err = io.Copyn(tr.r, ignoreWriter{}, nr);
	}
	tr.nb, tr.pad = 0, 0;
}

func (tr *Reader) verifyChecksum(header []byte) bool {
	given := tr.octal(header[148:156]);
	if tr.err != nil {
		return false
	}

	// POSIX specifies a sum of the unsigned byte values,
	// but the Sun tar uses signed byte values.  :-(
	var unsigned, signed int64;
	for i := 0; i < len(header); i++ {
		if i == 148 {
			// The chksum field is special: it should be treated as space bytes.
			unsigned += ' ' * 8;
			signed += ' ' * 8;
			i += 7;
			continue
		}
		unsigned += int64(header[i]);
		signed += int64(int8(header[i]));
	}

	return given == unsigned || given == signed
}

type slicer []byte
func (sp *slicer) next(n int) (b []byte) {
	s := *sp;
	b, *sp = s[0:n], s[n:len(s)];
	return
}

func (tr *Reader) readHeader() *Header {
	header := make([]byte, blockSize);
	var n int;
	if n, tr.err = io.ReadFull(tr.r, header); tr.err != nil {
		return nil
	}

	// Two blocks of zero bytes marks the end of the archive.
	if bytes.Equal(header, zeroBlock[0:blockSize]) {
		if n, tr.err = io.ReadFull(tr.r, header); tr.err != nil {
			return nil
		}
		if !bytes.Equal(header, zeroBlock[0:blockSize]) {
			tr.err = HeaderError;
		}
		return nil
	}

	if !tr.verifyChecksum(header) {
		tr.err = HeaderError;
		return nil
	}

	// Unpack
	hdr := new(Header);
	s := slicer(header);

	hdr.Name = cString(s.next(100));
	hdr.Mode = tr.octal(s.next(8));
	hdr.Uid = tr.octal(s.next(8));
	hdr.Gid = tr.octal(s.next(8));
	hdr.Size = tr.octal(s.next(12));
	hdr.Mtime = tr.octal(s.next(12));
	s.next(8);  // chksum
	hdr.Typeflag = s.next(1)[0];
	hdr.Linkname = cString(s.next(100));

	// The remainder of the header depends on the value of magic.
	// The original (v7) version of tar had no explicit magic field,
	// so its magic bytes, like the rest of the block, are NULs.
	magic := string(s.next(8));  // contains version field as well.
	var format string;
	switch magic {
	case "ustar\x0000":  // POSIX tar (1003.1-1988)
		if string(header[508:512]) == "tar\x00" {
			format = "star";
		} else {
			format = "posix";
		}
	case "ustar  \x00":  // old GNU tar
		format = "gnu";
	}

	switch format {
	case "posix", "gnu", "star":
		hdr.Uname = cString(s.next(32));
		hdr.Gname = cString(s.next(32));
		devmajor := s.next(8);
		devminor := s.next(8);
		if hdr.Typeflag == TypeChar || hdr.Typeflag == TypeBlock {
			hdr.Devmajor = tr.octal(devmajor);
			hdr.Devminor = tr.octal(devminor);
		}
		var prefix string;
		switch format {
		case "posix", "gnu":
			prefix = cString(s.next(155));
		case "star":
			prefix = cString(s.next(131));
			hdr.Atime = tr.octal(s.next(12));
			hdr.Ctime = tr.octal(s.next(12));
		}
		if len(prefix) > 0 {
			hdr.Name = prefix + "/" + hdr.Name;
		}
	}

	if tr.err != nil {
		tr.err = HeaderError;
		return nil
	}

	// Maximum value of hdr.Size is 64 GB (12 octal digits),
	// so there's no risk of int64 overflowing.
	tr.nb = int64(hdr.Size);
	tr.pad = -tr.nb & (blockSize - 1);  // blockSize is a power of two

	return hdr
}

// Read reads from the current entry in the tar archive.
// It returns 0, nil when it reaches the end of that entry,
// until Next is called to advance to the next entry.
func (tr *Reader) Read(b []uint8) (n int, err os.Error) {
	if int64(len(b)) > tr.nb {
		b = b[0:tr.nb];
	}
	n, err = tr.r.Read(b);
	tr.nb -= int64(n);
	tr.err = err;
	return
}
