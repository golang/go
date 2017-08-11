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
	"path"
	"sort"
	"strconv"
	"strings"
	"time"
)

var (
	ErrWriteTooLong    = errors.New("archive/tar: write too long")
	ErrFieldTooLong    = errors.New("archive/tar: header field too long")
	ErrWriteAfterClose = errors.New("archive/tar: write after close")
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
	usedBinary bool  // whether the binary numeric field extension was used
	preferPax  bool  // use PAX header instead of binary numeric header
	hdrBuff    block // buffer to use in writeHeader when writing a regular header
	paxHdrBuff block // buffer to use in writeHeader when writing a PAX header

	blk block // Buffer to use as temporary local storage
}

// NewWriter creates a new Writer writing to w.
func NewWriter(w io.Writer) *Writer { return &Writer{w: w} }

// Flush finishes writing the current file's block padding.
// The current file must be fully written before Flush can be called.
//
// Deprecated: This is unecessary as the next call to WriteHeader or Close
// will implicitly flush out the file's padding.
func (tw *Writer) Flush() error {
	if tw.nb > 0 {
		tw.err = fmt.Errorf("archive/tar: missed writing %d bytes", tw.nb)
		return tw.err
	}
	tw.err = tw.writePadding()
	return tw.err
}

func (tw *Writer) writePadding() error {
	if _, err := tw.w.Write(zeroBlock[:tw.pad]); err != nil {
		return err
	}
	tw.pad = 0
	return nil
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
	// TODO(dsnet): Add PAX timestamps with nanosecond support.
	hdrCpy := *hdr
	hdrCpy.ModTime = hdrCpy.ModTime.Truncate(time.Second)

	switch allowedFormats, _ := hdrCpy.allowedFormats(); {
	case allowedFormats&formatUSTAR != 0:
		return tw.writeUSTARHeader(&hdrCpy)
	case allowedFormats&formatPAX != 0:
		// TODO(dsnet): Implement and call specialized writePAXHeader.
		return tw.writeHeader(&hdrCpy, true)
	case allowedFormats&formatGNU != 0:
		// TODO(dsnet): Implement and call specialized writeGNUHeader.
		return tw.writeHeader(&hdrCpy, true)
	default:
		return ErrHeader
	}
}

func (tw *Writer) writeUSTARHeader(hdr *Header) error {
	// TODO(dsnet): Support USTAR prefix/suffix path splitting.
	// See https://golang.org/issue/12594

	// Pack the main header.
	var f formatter
	blk := tw.templateV7Plus(hdr, &f)
	blk.SetFormat(formatUSTAR)
	if f.err != nil {
		return f.err // Should never happen since header is validated
	}
	return tw.writeRawHeader(blk, hdr.Size)
}

// templateV7Plus fills out the V7 fields of a block using values from hdr.
// It also fills out fields (uname, gname, devmajor, devminor) that are
// shared in the USTAR, PAX, and GNU formats.
//
// The block returned is only valid until the next call to templateV7Plus.
func (tw *Writer) templateV7Plus(hdr *Header, f *formatter) *block {
	tw.blk.Reset()

	modTime := hdr.ModTime
	if modTime.IsZero() {
		modTime = time.Unix(0, 0)
	}

	v7 := tw.blk.V7()
	v7.TypeFlag()[0] = hdr.Typeflag
	f.formatString(v7.Name(), hdr.Name)
	f.formatString(v7.LinkName(), hdr.Linkname)
	f.formatOctal(v7.Mode(), hdr.Mode)
	f.formatOctal(v7.UID(), int64(hdr.Uid))
	f.formatOctal(v7.GID(), int64(hdr.Gid))
	f.formatOctal(v7.Size(), hdr.Size)
	f.formatOctal(v7.ModTime(), modTime.Unix())

	ustar := tw.blk.USTAR()
	f.formatString(ustar.UserName(), hdr.Uname)
	f.formatString(ustar.GroupName(), hdr.Gname)
	f.formatOctal(ustar.DevMajor(), hdr.Devmajor)
	f.formatOctal(ustar.DevMinor(), hdr.Devminor)

	return &tw.blk
}

// writeRawHeader writes the value of blk, regardless of its value.
// It sets up the Writer such that it can accept a file of the given size.
func (tw *Writer) writeRawHeader(blk *block, size int64) error {
	if err := tw.Flush(); err != nil {
		return err
	}
	if _, err := tw.w.Write(blk[:]); err != nil {
		return err
	}
	// TODO(dsnet): Set Size implicitly to zero for header-only entries.
	// See https://golang.org/issue/15565
	tw.nb = size
	tw.pad = -size & (blockSize - 1) // blockSize is a power of two
	return nil
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

	// TODO(dsnet): we might want to use PAX headers for
	// subsecond time resolution, but for now let's just capture
	// too long fields or non ascii characters

	// We need to select which scratch buffer to use carefully,
	// since this method is called recursively to write PAX headers.
	// If allowPax is true, this is the non-recursive call, and we will use hdrBuff.
	// If allowPax is false, we are being called by writePAXHeader, and hdrBuff is
	// already being used by the non-recursive call, so we must use paxHdrBuff.
	header := &tw.hdrBuff
	if !allowPax {
		header = &tw.paxHdrBuff
	}
	copy(header[:], zeroBlock[:])

	// Wrappers around formatter that automatically sets paxHeaders if the
	// argument extends beyond the capacity of the input byte slice.
	var f formatter
	var formatString = func(b []byte, s string, paxKeyword string) {
		needsPaxHeader := paxKeyword != paxNone && len(s) > len(b) || !isASCII(s)
		if needsPaxHeader {
			paxHeaders[paxKeyword] = s
		}

		// Write string in a best-effort manner to satisfy readers that expect
		// the field to be non-empty.
		s = toASCII(s)
		if len(s) > len(b) {
			s = s[:len(b)]
		}
		f.formatString(b, s) // Should never error
	}
	var formatNumeric = func(b []byte, x int64, paxKeyword string) {
		if !fitsInOctal(len(b), x) {
			if paxKeyword != paxNone && tw.preferPax {
				// Use PAX format.
				f.formatOctal(b, 0)
				paxHeaders[paxKeyword] = strconv.FormatInt(x, 10)
				return
			} else {
				// Use GNU format.
				tw.usedBinary = true
			}
		}
		f.formatNumeric(b, x)
	}

	// Handle out of range ModTime carefully.
	var modTime int64
	if !hdr.ModTime.Before(minTime) && !hdr.ModTime.After(maxTime) {
		modTime = hdr.ModTime.Unix()
	}

	v7 := header.V7()
	formatString(v7.Name(), hdr.Name, paxPath)
	// TODO(dsnet): The GNU format permits the mode field to be encoded in
	// base-256 format. Thus, we can use formatNumeric instead of formatOctal.
	f.formatOctal(v7.Mode(), hdr.Mode)
	formatNumeric(v7.UID(), int64(hdr.Uid), paxUid)
	formatNumeric(v7.GID(), int64(hdr.Gid), paxGid)
	formatNumeric(v7.Size(), hdr.Size, paxSize)
	// TODO(dsnet): Consider using PAX for finer time granularity.
	formatNumeric(v7.ModTime(), modTime, paxNone)
	v7.TypeFlag()[0] = hdr.Typeflag
	formatString(v7.LinkName(), hdr.Linkname, paxLinkpath)

	ustar := header.USTAR()
	formatString(ustar.UserName(), hdr.Uname, paxUname)
	formatString(ustar.GroupName(), hdr.Gname, paxGname)
	formatNumeric(ustar.DevMajor(), hdr.Devmajor, paxNone)
	formatNumeric(ustar.DevMinor(), hdr.Devminor, paxNone)

	// TODO(dsnet): The logic surrounding the prefix field is broken when trying
	// to encode the header as GNU format. The challenge with the current logic
	// is that we are unsure what format we are using at any given moment until
	// we have processed *all* of the fields. The problem is that by the time
	// all fields have been processed, some work has already been done to handle
	// each field under the assumption that it is for one given format or
	// another. In some situations, this causes the Writer to be confused and
	// encode a prefix field when the format being used is GNU. Thus, producing
	// an invalid tar file.
	//
	// As a short-term fix, we disable the logic to use the prefix field, which
	// will force the badly generated GNU files to become encoded as being
	// the PAX format.
	//
	// As an alternative fix, we could hard-code preferPax to be true. However,
	// this is problematic for the following reasons:
	//	* The preferPax functionality is not tested at all.
	//	* This can result in headers that try to use both the GNU and PAX
	//	features at the same time, which is also wrong.
	//
	// The proper fix for this is to use a two-pass method:
	//	* The first pass simply determines what set of formats can possibly
	//	encode the given header.
	//	* The second pass actually encodes the header as that given format
	//	without worrying about violating the format.
	//
	// See the following:
	//	https://golang.org/issue/12594
	//	https://golang.org/issue/17630
	//	https://golang.org/issue/9683
	const usePrefix = false

	// try to use a ustar header when only the name is too long
	_, paxPathUsed := paxHeaders[paxPath]
	if usePrefix && !tw.preferPax && len(paxHeaders) == 1 && paxPathUsed {
		prefix, suffix, ok := splitUSTARPath(hdr.Name)
		if ok {
			// Since we can encode in USTAR format, disable PAX header.
			delete(paxHeaders, paxPath)

			// Update the path fields
			formatString(v7.Name(), suffix, paxNone)
			formatString(ustar.Prefix(), prefix, paxNone)
		}
	}

	if tw.usedBinary {
		header.SetFormat(formatGNU)
	} else {
		header.SetFormat(formatUSTAR)
	}

	// Check if there were any formatting errors.
	if f.err != nil {
		tw.err = f.err
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
	tw.nb = hdr.Size
	tw.pad = (blockSize - (tw.nb % blockSize)) % blockSize

	_, tw.err = tw.w.Write(header[:])
	return tw.err
}

// splitUSTARPath splits a path according to USTAR prefix and suffix rules.
// If the path is not splittable, then it will return ("", "", false).
func splitUSTARPath(name string) (prefix, suffix string, ok bool) {
	length := len(name)
	if length <= nameSize || !isASCII(name) {
		return "", "", false
	} else if length > prefixSize+1 {
		length = prefixSize + 1
	} else if name[length-1] == '/' {
		length--
	}

	i := strings.LastIndex(name[:length], "/")
	nlen := len(name) - i - 1 // nlen is length of suffix
	plen := i                 // plen is length of prefix
	if i <= 0 || nlen > nameSize || nlen == 0 || plen > prefixSize {
		return "", "", false
	}
	return name[:i], name[i+1:], true
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
	// with the current pid. However, this results in differing outputs
	// for identical inputs. As such, the constant 0 is now used instead.
	// golang.org/issue/12358
	dir, file := path.Split(hdr.Name)
	fullName := path.Join(dir, "PaxHeaders.0", file)

	ascii := toASCII(fullName)
	if len(ascii) > nameSize {
		ascii = ascii[:nameSize]
	}
	ext.Name = ascii
	// Construct the body
	var buf bytes.Buffer

	// Keys are sorted before writing to body to allow deterministic output.
	keys := make([]string, 0, len(paxHeaders))
	for k := range paxHeaders {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		rec, err := formatPAXRecord(k, paxHeaders[k])
		if err != nil {
			return err
		}
		fmt.Fprint(&buf, rec)
	}

	ext.Size = int64(len(buf.Bytes()))
	if err := tw.writeHeader(ext, false); err != nil {
		return err
	}
	if _, err := tw.Write(buf.Bytes()); err != nil {
		return err
	}
	return tw.writePadding()
}

// Write writes to the current entry in the tar archive.
// Write returns the error ErrWriteTooLong if more than
// hdr.Size bytes are written after WriteHeader.
func (tw *Writer) Write(b []byte) (n int, err error) {
	if tw.closed {
		err = ErrWriteAfterClose
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
		_, tw.err = tw.w.Write(zeroBlock[:])
		if tw.err != nil {
			break
		}
	}
	return tw.err
}
