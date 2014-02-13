// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

// TODO(dsymonds):
// - catch more errors (no first header, etc.)

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"strconv"
	"strings"
	"time"
)

var (
	ErrWriteTooLong    = errors.New("archive/tar: write too long")
	ErrFieldTooLong    = errors.New("archive/tar: header field too long")
	ErrWriteAfterClose = errors.New("archive/tar: write after close")
	errNameTooLong     = errors.New("archive/tar: name too long")
	errInvalidHeader   = errors.New("archive/tar: header field too long or contains invalid values")
)

// A Writer provides sequential writing of a tar archive in POSIX.1 format.
// A tar archive consists of a sequence of files.
// Call WriteHeader to begin a new file, and then call Write to supply that file's data,
// writing at most hdr.Size bytes in total.
type Writer struct {
	w          io.Writer
	err        error
	nb         int64 // number of unwritten bytes for current file entry
	pad        int64 // amount of padding to write after current file entry
	closed     bool
	usedBinary bool // whether the binary numeric field extension was used
	preferPax  bool // use pax header instead of binary numeric header
}

// NewWriter creates a new Writer writing to w.
func NewWriter(w io.Writer) *Writer { return &Writer{w: w} }

// Flush finishes writing the current file (optional).
func (tw *Writer) Flush() error {
	if tw.nb > 0 {
		tw.err = fmt.Errorf("archive/tar: missed writing %d bytes", tw.nb)
		return tw.err
	}

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
// If the value is too long for the field and allowPax is true add a paxheader record instead
func (tw *Writer) cString(b []byte, s string, allowPax bool, paxKeyword string, paxHeaders map[string]string) {
	needsPaxHeader := allowPax && len(s) > len(b) || !isASCII(s)
	if needsPaxHeader {
		paxHeaders[paxKeyword] = s
		return
	}
	if len(s) > len(b) {
		if tw.err == nil {
			tw.err = ErrFieldTooLong
		}
		return
	}
	ascii := toASCII(s)
	copy(b, ascii)
	if len(ascii) < len(b) {
		b[len(ascii)] = 0
	}
}

// Encode x as an octal ASCII string and write it into b with leading zeros.
func (tw *Writer) octal(b []byte, x int64) {
	s := strconv.FormatInt(x, 8)
	// leading zeros, but leave room for a NUL.
	for len(s)+1 < len(b) {
		s = "0" + s
	}
	tw.cString(b, s, false, paxNone, nil)
}

// Write x into b, either as octal or as binary (GNUtar/star extension).
// If the value is too long for the field and writingPax is enabled both for the field and the add a paxheader record instead
func (tw *Writer) numeric(b []byte, x int64, allowPax bool, paxKeyword string, paxHeaders map[string]string) {
	// Try octal first.
	s := strconv.FormatInt(x, 8)
	if len(s) < len(b) {
		tw.octal(b, x)
		return
	}

	// If it is too long for octal, and pax is preferred, use a pax header
	if allowPax && tw.preferPax {
		tw.octal(b, 0)
		s := strconv.FormatInt(x, 10)
		paxHeaders[paxKeyword] = s
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

var (
	minTime = time.Unix(0, 0)
	// There is room for 11 octal digits (33 bits) of mtime.
	maxTime = minTime.Add((1<<33 - 1) * time.Second)
)

// WriteHeader writes hdr and prepares to accept the file's contents.
// WriteHeader calls Flush if it is not the first header.
// Calling after a Close will return ErrWriteAfterClose.
func (tw *Writer) WriteHeader(hdr *Header) error {
	return tw.writeHeader(hdr, true)
}

// WriteHeader writes hdr and prepares to accept the file's contents.
// WriteHeader calls Flush if it is not the first header.
// Calling after a Close will return ErrWriteAfterClose.
// As this method is called internally by writePax header to allow it to
// suppress writing the pax header.
func (tw *Writer) writeHeader(hdr *Header, allowPax bool) error {
	if tw.closed {
		return ErrWriteAfterClose
	}
	if tw.err == nil {
		tw.Flush()
	}
	if tw.err != nil {
		return tw.err
	}

	// a map to hold pax header records, if any are needed
	paxHeaders := make(map[string]string)

	// TODO(shanemhansen): we might want to use PAX headers for
	// subsecond time resolution, but for now let's just capture
	// too long fields or non ascii characters

	header := make([]byte, blockSize)
	s := slicer(header)

	// keep a reference to the filename to allow to overwrite it later if we detect that we can use ustar longnames instead of pax
	pathHeaderBytes := s.next(fileNameSize)

	tw.cString(pathHeaderBytes, hdr.Name, true, paxPath, paxHeaders)

	// Handle out of range ModTime carefully.
	var modTime int64
	if !hdr.ModTime.Before(minTime) && !hdr.ModTime.After(maxTime) {
		modTime = hdr.ModTime.Unix()
	}

	tw.octal(s.next(8), hdr.Mode)                                   // 100:108
	tw.numeric(s.next(8), int64(hdr.Uid), true, paxUid, paxHeaders) // 108:116
	tw.numeric(s.next(8), int64(hdr.Gid), true, paxGid, paxHeaders) // 116:124
	tw.numeric(s.next(12), hdr.Size, true, paxSize, paxHeaders)     // 124:136
	tw.numeric(s.next(12), modTime, false, paxNone, nil)            // 136:148 --- consider using pax for finer granularity
	s.next(8)                                                       // chksum (148:156)
	s.next(1)[0] = hdr.Typeflag                                     // 156:157

	tw.cString(s.next(100), hdr.Linkname, true, paxLinkpath, paxHeaders)

	copy(s.next(8), []byte("ustar\x0000"))                        // 257:265
	tw.cString(s.next(32), hdr.Uname, true, paxUname, paxHeaders) // 265:297
	tw.cString(s.next(32), hdr.Gname, true, paxGname, paxHeaders) // 297:329
	tw.numeric(s.next(8), hdr.Devmajor, false, paxNone, nil)      // 329:337
	tw.numeric(s.next(8), hdr.Devminor, false, paxNone, nil)      // 337:345

	// keep a reference to the prefix to allow to overwrite it later if we detect that we can use ustar longnames instead of pax
	prefixHeaderBytes := s.next(155)
	tw.cString(prefixHeaderBytes, "", false, paxNone, nil) // 345:500  prefix

	// Use the GNU magic instead of POSIX magic if we used any GNU extensions.
	if tw.usedBinary {
		copy(header[257:265], []byte("ustar  \x00"))
	}

	_, paxPathUsed := paxHeaders[paxPath]
	// try to use a ustar header when only the name is too long
	if !tw.preferPax && len(paxHeaders) == 1 && paxPathUsed {
		suffix := hdr.Name
		prefix := ""
		if len(hdr.Name) > fileNameSize && isASCII(hdr.Name) {
			var err error
			prefix, suffix, err = tw.splitUSTARLongName(hdr.Name)
			if err == nil {
				// ok we can use a ustar long name instead of pax, now correct the fields

				// remove the path field from the pax header. this will suppress the pax header
				delete(paxHeaders, paxPath)

				// update the path fields
				tw.cString(pathHeaderBytes, suffix, false, paxNone, nil)
				tw.cString(prefixHeaderBytes, prefix, false, paxNone, nil)

				// Use the ustar magic if we used ustar long names.
				if len(prefix) > 0 {
					copy(header[257:265], []byte("ustar\000"))
				}
			}
		}
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

	if allowPax {
		for k, v := range hdr.Xattrs {
			paxHeaders[paxXattr+k] = v
		}
	}

	if len(paxHeaders) > 0 {
		if !allowPax {
			return errInvalidHeader
		}
		if err := tw.writePAXHeader(hdr, paxHeaders); err != nil {
			return err
		}
	}
	tw.nb = int64(hdr.Size)
	tw.pad = (blockSize - (tw.nb % blockSize)) % blockSize

	_, tw.err = tw.w.Write(header)
	return tw.err
}

// writeUSTARLongName splits a USTAR long name hdr.Name.
// name must be < 256 characters. errNameTooLong is returned
// if hdr.Name can't be split. The splitting heuristic
// is compatible with gnu tar.
func (tw *Writer) splitUSTARLongName(name string) (prefix, suffix string, err error) {
	length := len(name)
	if length > fileNamePrefixSize+1 {
		length = fileNamePrefixSize + 1
	} else if name[length-1] == '/' {
		length--
	}
	i := strings.LastIndex(name[:length], "/")
	// nlen contains the resulting length in the name field.
	// plen contains the resulting length in the prefix field.
	nlen := len(name) - i - 1
	plen := i
	if i <= 0 || nlen > fileNameSize || nlen == 0 || plen > fileNamePrefixSize {
		err = errNameTooLong
		return
	}
	prefix, suffix = name[:i], name[i+1:]
	return
}

// writePaxHeader writes an extended pax header to the
// archive.
func (tw *Writer) writePAXHeader(hdr *Header, paxHeaders map[string]string) error {
	// Prepare extended header
	ext := new(Header)
	ext.Typeflag = TypeXHeader
	// Setting ModTime is required for reader parsing to
	// succeed, and seems harmless enough.
	ext.ModTime = hdr.ModTime
	// The spec asks that we namespace our pseudo files
	// with the current pid.
	pid := os.Getpid()
	dir, file := path.Split(hdr.Name)
	fullName := path.Join(dir,
		fmt.Sprintf("PaxHeaders.%d", pid), file)

	ascii := toASCII(fullName)
	if len(ascii) > 100 {
		ascii = ascii[:100]
	}
	ext.Name = ascii
	// Construct the body
	var buf bytes.Buffer

	for k, v := range paxHeaders {
		fmt.Fprint(&buf, paxHeader(k+"="+v))
	}

	ext.Size = int64(len(buf.Bytes()))
	if err := tw.writeHeader(ext, false); err != nil {
		return err
	}
	if _, err := tw.Write(buf.Bytes()); err != nil {
		return err
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	return nil
}

// paxHeader formats a single pax record, prefixing it with the appropriate length
func paxHeader(msg string) string {
	const padding = 2 // Extra padding for space and newline
	size := len(msg) + padding
	size += len(strconv.Itoa(size))
	record := fmt.Sprintf("%d %s\n", size, msg)
	if len(record) != size {
		// Final adjustment if adding size increased
		// the number of digits in size
		size = len(record)
		record = fmt.Sprintf("%d %s\n", size, msg)
	}
	return record
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
	if tw.err != nil {
		return tw.err
	}

	// trailer: two zero blocks
	for i := 0; i < 2; i++ {
		_, tw.err = tw.w.Write(zeroBlock)
		if tw.err != nil {
			break
		}
	}
	return tw.err
}
