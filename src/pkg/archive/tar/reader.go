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
	r    io.Reader
	err  error
	pad  int64          // amount of padding (ignored) after current file entry
	curr numBytesReader // reader for current file entry
}

// A numBytesReader is an io.Reader with a numBytes method, returning the number
// of bytes remaining in the underlying encoded data.
type numBytesReader interface {
	io.Reader
	numBytes() int64
}

// A regFileReader is a numBytesReader for reading file data from a tar archive.
type regFileReader struct {
	r  io.Reader // underlying reader
	nb int64     // number of unread bytes for current file entry
}

// A sparseFileReader is a numBytesReader for reading sparse file data from a tar archive.
type sparseFileReader struct {
	rfr *regFileReader // reads the sparse-encoded file data
	sp  []sparseEntry  // the sparse map for the file
	pos int64          // keeps track of file position
	tot int64          // total size of the file
}

// Keywords for GNU sparse files in a PAX extended header
const (
	paxGNUSparseNumBlocks = "GNU.sparse.numblocks"
	paxGNUSparseOffset    = "GNU.sparse.offset"
	paxGNUSparseNumBytes  = "GNU.sparse.numbytes"
	paxGNUSparseMap       = "GNU.sparse.map"
	paxGNUSparseName      = "GNU.sparse.name"
	paxGNUSparseMajor     = "GNU.sparse.major"
	paxGNUSparseMinor     = "GNU.sparse.minor"
	paxGNUSparseSize      = "GNU.sparse.size"
	paxGNUSparseRealSize  = "GNU.sparse.realsize"
)

// Keywords for old GNU sparse headers
const (
	oldGNUSparseMainHeaderOffset               = 386
	oldGNUSparseMainHeaderIsExtendedOffset     = 482
	oldGNUSparseMainHeaderNumEntries           = 4
	oldGNUSparseExtendedHeaderIsExtendedOffset = 504
	oldGNUSparseExtendedHeaderNumEntries       = 21
	oldGNUSparseOffsetSize                     = 12
	oldGNUSparseNumBytesSize                   = 12
)

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

		// Check for a PAX format sparse file
		sp, err := tr.checkForGNUSparsePAXHeaders(hdr, headers)
		if err != nil {
			tr.err = err
			return nil, err
		}
		if sp != nil {
			// Current file is a PAX format GNU sparse file.
			// Set the current file reader to a sparse file reader.
			tr.curr = &sparseFileReader{rfr: tr.curr.(*regFileReader), sp: sp, tot: hdr.Size}
		}
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

// checkForGNUSparsePAXHeaders checks the PAX headers for GNU sparse headers. If they are found, then
// this function reads the sparse map and returns it. Unknown sparse formats are ignored, causing the file to
// be treated as a regular file.
func (tr *Reader) checkForGNUSparsePAXHeaders(hdr *Header, headers map[string]string) ([]sparseEntry, error) {
	var sparseFormat string

	// Check for sparse format indicators
	major, majorOk := headers[paxGNUSparseMajor]
	minor, minorOk := headers[paxGNUSparseMinor]
	sparseName, sparseNameOk := headers[paxGNUSparseName]
	_, sparseMapOk := headers[paxGNUSparseMap]
	sparseSize, sparseSizeOk := headers[paxGNUSparseSize]
	sparseRealSize, sparseRealSizeOk := headers[paxGNUSparseRealSize]

	// Identify which, if any, sparse format applies from which PAX headers are set
	if majorOk && minorOk {
		sparseFormat = major + "." + minor
	} else if sparseNameOk && sparseMapOk {
		sparseFormat = "0.1"
	} else if sparseSizeOk {
		sparseFormat = "0.0"
	} else {
		// Not a PAX format GNU sparse file.
		return nil, nil
	}

	// Check for unknown sparse format
	if sparseFormat != "0.0" && sparseFormat != "0.1" && sparseFormat != "1.0" {
		return nil, nil
	}

	// Update hdr from GNU sparse PAX headers
	if sparseNameOk {
		hdr.Name = sparseName
	}
	if sparseSizeOk {
		realSize, err := strconv.ParseInt(sparseSize, 10, 0)
		if err != nil {
			return nil, ErrHeader
		}
		hdr.Size = realSize
	} else if sparseRealSizeOk {
		realSize, err := strconv.ParseInt(sparseRealSize, 10, 0)
		if err != nil {
			return nil, ErrHeader
		}
		hdr.Size = realSize
	}

	// Set up the sparse map, according to the particular sparse format in use
	var sp []sparseEntry
	var err error
	switch sparseFormat {
	case "0.0", "0.1":
		sp, err = readGNUSparseMap0x1(headers)
	case "1.0":
		sp, err = readGNUSparseMap1x0(tr.curr)
	}
	return sp, err
}

// mergePAX merges well known headers according to PAX standard.
// In general headers with the same name as those found
// in the header struct overwrite those found in the header
// struct with higher precision or longer values. Esp. useful
// for name and linkname fields.
func mergePAX(hdr *Header, headers map[string]string) error {
	for k, v := range headers {
		switch k {
		case paxPath:
			hdr.Name = v
		case paxLinkpath:
			hdr.Linkname = v
		case paxGname:
			hdr.Gname = v
		case paxUname:
			hdr.Uname = v
		case paxUid:
			uid, err := strconv.ParseInt(v, 10, 0)
			if err != nil {
				return err
			}
			hdr.Uid = int(uid)
		case paxGid:
			gid, err := strconv.ParseInt(v, 10, 0)
			if err != nil {
				return err
			}
			hdr.Gid = int(gid)
		case paxAtime:
			t, err := parsePAXTime(v)
			if err != nil {
				return err
			}
			hdr.AccessTime = t
		case paxMtime:
			t, err := parsePAXTime(v)
			if err != nil {
				return err
			}
			hdr.ModTime = t
		case paxCtime:
			t, err := parsePAXTime(v)
			if err != nil {
				return err
			}
			hdr.ChangeTime = t
		case paxSize:
			size, err := strconv.ParseInt(v, 10, 0)
			if err != nil {
				return err
			}
			hdr.Size = int64(size)
		default:
			if strings.HasPrefix(k, paxXattr) {
				if hdr.Xattrs == nil {
					hdr.Xattrs = make(map[string]string)
				}
				hdr.Xattrs[k[len(paxXattr):]] = v
			}
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

	// For GNU PAX sparse format 0.0 support.
	// This function transforms the sparse format 0.0 headers into sparse format 0.1 headers.
	var sparseMap bytes.Buffer

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
		// beginning to eat the ' ', -1 on the end to skip the newline.
		var record []byte
		record, buf = buf[sp+1:n-1], buf[n:]
		// The first equals is guaranteed to mark the end of the key.
		// Everything else is value.
		eq := bytes.IndexByte(record, '=')
		if eq == -1 {
			return nil, ErrHeader
		}
		key, value := record[:eq], record[eq+1:]

		keyStr := string(key)
		if keyStr == paxGNUSparseOffset || keyStr == paxGNUSparseNumBytes {
			// GNU sparse format 0.0 special key. Write to sparseMap instead of using the headers map.
			sparseMap.Write(value)
			sparseMap.Write([]byte{','})
		} else {
			// Normal key. Set the value in the headers map.
			headers[keyStr] = string(value)
		}
	}
	if sparseMap.Len() != 0 {
		// Add sparse info to headers, chopping off the extra comma
		sparseMap.Truncate(sparseMap.Len() - 1)
		headers[paxGNUSparseMap] = sparseMap.String()
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

	// Because unused fields are filled with NULs, we need
	// to skip leading NULs. Fields may also be padded with
	// spaces or NULs.
	// So we remove leading and trailing NULs and spaces to
	// be sure.
	b = bytes.Trim(b, " \x00")

	if len(b) == 0 {
		return 0
	}
	x, err := strconv.ParseUint(cString(b), 8, 64)
	if err != nil {
		tr.err = err
	}
	return int64(x)
}

// skipUnread skips any unread bytes in the existing file entry, as well as any alignment padding.
func (tr *Reader) skipUnread() {
	nr := tr.numBytes() + tr.pad // number of bytes to skip
	tr.curr, tr.pad = nil, 0
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
	switch {
	case magic[:6] == "ustar\x00": // POSIX tar (1003.1-1988)
		if string(header[508:512]) == "tar\x00" {
			format = "star"
		} else {
			format = "posix"
		}
	case magic == "ustar  \x00": // old GNU tar
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
	nb := int64(hdr.Size)
	tr.pad = -nb & (blockSize - 1) // blockSize is a power of two

	// Set the current file reader.
	tr.curr = &regFileReader{r: tr.r, nb: nb}

	// Check for old GNU sparse format entry.
	if hdr.Typeflag == TypeGNUSparse {
		// Get the real size of the file.
		hdr.Size = tr.octal(header[483:495])

		// Read the sparse map.
		sp := tr.readOldGNUSparseMap(header)
		if tr.err != nil {
			return nil
		}
		// Current file is a GNU sparse file. Update the current file reader.
		tr.curr = &sparseFileReader{rfr: tr.curr.(*regFileReader), sp: sp, tot: hdr.Size}
	}

	return hdr
}

// A sparseEntry holds a single entry in a sparse file's sparse map.
// A sparse entry indicates the offset and size in a sparse file of a
// block of data.
type sparseEntry struct {
	offset   int64
	numBytes int64
}

// readOldGNUSparseMap reads the sparse map as stored in the old GNU sparse format.
// The sparse map is stored in the tar header if it's small enough. If it's larger than four entries,
// then one or more extension headers are used to store the rest of the sparse map.
func (tr *Reader) readOldGNUSparseMap(header []byte) []sparseEntry {
	isExtended := header[oldGNUSparseMainHeaderIsExtendedOffset] != 0
	spCap := oldGNUSparseMainHeaderNumEntries
	if isExtended {
		spCap += oldGNUSparseExtendedHeaderNumEntries
	}
	sp := make([]sparseEntry, 0, spCap)
	s := slicer(header[oldGNUSparseMainHeaderOffset:])

	// Read the four entries from the main tar header
	for i := 0; i < oldGNUSparseMainHeaderNumEntries; i++ {
		offset := tr.octal(s.next(oldGNUSparseOffsetSize))
		numBytes := tr.octal(s.next(oldGNUSparseNumBytesSize))
		if tr.err != nil {
			tr.err = ErrHeader
			return nil
		}
		if offset == 0 && numBytes == 0 {
			break
		}
		sp = append(sp, sparseEntry{offset: offset, numBytes: numBytes})
	}

	for isExtended {
		// There are more entries. Read an extension header and parse its entries.
		sparseHeader := make([]byte, blockSize)
		if _, tr.err = io.ReadFull(tr.r, sparseHeader); tr.err != nil {
			return nil
		}
		isExtended = sparseHeader[oldGNUSparseExtendedHeaderIsExtendedOffset] != 0
		s = slicer(sparseHeader)
		for i := 0; i < oldGNUSparseExtendedHeaderNumEntries; i++ {
			offset := tr.octal(s.next(oldGNUSparseOffsetSize))
			numBytes := tr.octal(s.next(oldGNUSparseNumBytesSize))
			if tr.err != nil {
				tr.err = ErrHeader
				return nil
			}
			if offset == 0 && numBytes == 0 {
				break
			}
			sp = append(sp, sparseEntry{offset: offset, numBytes: numBytes})
		}
	}
	return sp
}

// readGNUSparseMap1x0 reads the sparse map as stored in GNU's PAX sparse format version 1.0.
// The sparse map is stored just before the file data and padded out to the nearest block boundary.
func readGNUSparseMap1x0(r io.Reader) ([]sparseEntry, error) {
	buf := make([]byte, 2*blockSize)
	sparseHeader := buf[:blockSize]

	// readDecimal is a helper function to read a decimal integer from the sparse map
	// while making sure to read from the file in blocks of size blockSize
	readDecimal := func() (int64, error) {
		// Look for newline
		nl := bytes.IndexByte(sparseHeader, '\n')
		if nl == -1 {
			if len(sparseHeader) >= blockSize {
				// This is an error
				return 0, ErrHeader
			}
			oldLen := len(sparseHeader)
			newLen := oldLen + blockSize
			if cap(sparseHeader) < newLen {
				// There's more header, but we need to make room for the next block
				copy(buf, sparseHeader)
				sparseHeader = buf[:newLen]
			} else {
				// There's more header, and we can just reslice
				sparseHeader = sparseHeader[:newLen]
			}

			// Now that sparseHeader is large enough, read next block
			if _, err := io.ReadFull(r, sparseHeader[oldLen:newLen]); err != nil {
				return 0, err
			}

			// Look for a newline in the new data
			nl = bytes.IndexByte(sparseHeader[oldLen:newLen], '\n')
			if nl == -1 {
				// This is an error
				return 0, ErrHeader
			}
			nl += oldLen // We want the position from the beginning
		}
		// Now that we've found a newline, read a number
		n, err := strconv.ParseInt(string(sparseHeader[:nl]), 10, 0)
		if err != nil {
			return 0, ErrHeader
		}

		// Update sparseHeader to consume this number
		sparseHeader = sparseHeader[nl+1:]
		return n, nil
	}

	// Read the first block
	if _, err := io.ReadFull(r, sparseHeader); err != nil {
		return nil, err
	}

	// The first line contains the number of entries
	numEntries, err := readDecimal()
	if err != nil {
		return nil, err
	}

	// Read all the entries
	sp := make([]sparseEntry, 0, numEntries)
	for i := int64(0); i < numEntries; i++ {
		// Read the offset
		offset, err := readDecimal()
		if err != nil {
			return nil, err
		}
		// Read numBytes
		numBytes, err := readDecimal()
		if err != nil {
			return nil, err
		}

		sp = append(sp, sparseEntry{offset: offset, numBytes: numBytes})
	}

	return sp, nil
}

// readGNUSparseMap0x1 reads the sparse map as stored in GNU's PAX sparse format version 0.1.
// The sparse map is stored in the PAX headers.
func readGNUSparseMap0x1(headers map[string]string) ([]sparseEntry, error) {
	// Get number of entries
	numEntriesStr, ok := headers[paxGNUSparseNumBlocks]
	if !ok {
		return nil, ErrHeader
	}
	numEntries, err := strconv.ParseInt(numEntriesStr, 10, 0)
	if err != nil {
		return nil, ErrHeader
	}

	sparseMap := strings.Split(headers[paxGNUSparseMap], ",")

	// There should be two numbers in sparseMap for each entry
	if int64(len(sparseMap)) != 2*numEntries {
		return nil, ErrHeader
	}

	// Loop through the entries in the sparse map
	sp := make([]sparseEntry, 0, numEntries)
	for i := int64(0); i < numEntries; i++ {
		offset, err := strconv.ParseInt(sparseMap[2*i], 10, 0)
		if err != nil {
			return nil, ErrHeader
		}
		numBytes, err := strconv.ParseInt(sparseMap[2*i+1], 10, 0)
		if err != nil {
			return nil, ErrHeader
		}
		sp = append(sp, sparseEntry{offset: offset, numBytes: numBytes})
	}

	return sp, nil
}

// numBytes returns the number of bytes left to read in the current file's entry
// in the tar archive, or 0 if there is no current file.
func (tr *Reader) numBytes() int64 {
	if tr.curr == nil {
		// No current file, so no bytes
		return 0
	}
	return tr.curr.numBytes()
}

// Read reads from the current entry in the tar archive.
// It returns 0, io.EOF when it reaches the end of that entry,
// until Next is called to advance to the next entry.
func (tr *Reader) Read(b []byte) (n int, err error) {
	n, err = tr.curr.Read(b)
	if err != nil && err != io.EOF {
		tr.err = err
	}
	return
}

func (rfr *regFileReader) Read(b []byte) (n int, err error) {
	if rfr.nb == 0 {
		// file consumed
		return 0, io.EOF
	}
	if int64(len(b)) > rfr.nb {
		b = b[0:rfr.nb]
	}
	n, err = rfr.r.Read(b)
	rfr.nb -= int64(n)

	if err == io.EOF && rfr.nb > 0 {
		err = io.ErrUnexpectedEOF
	}
	return
}

// numBytes returns the number of bytes left to read in the file's data in the tar archive.
func (rfr *regFileReader) numBytes() int64 {
	return rfr.nb
}

// readHole reads a sparse file hole ending at offset toOffset
func (sfr *sparseFileReader) readHole(b []byte, toOffset int64) int {
	n64 := toOffset - sfr.pos
	if n64 > int64(len(b)) {
		n64 = int64(len(b))
	}
	n := int(n64)
	for i := 0; i < n; i++ {
		b[i] = 0
	}
	sfr.pos += n64
	return n
}

// Read reads the sparse file data in expanded form.
func (sfr *sparseFileReader) Read(b []byte) (n int, err error) {
	if len(sfr.sp) == 0 {
		// No more data fragments to read from.
		if sfr.pos < sfr.tot {
			// We're in the last hole
			n = sfr.readHole(b, sfr.tot)
			return
		}
		// Otherwise, we're at the end of the file
		return 0, io.EOF
	}
	if sfr.pos < sfr.sp[0].offset {
		// We're in a hole
		n = sfr.readHole(b, sfr.sp[0].offset)
		return
	}

	// We're not in a hole, so we'll read from the next data fragment
	posInFragment := sfr.pos - sfr.sp[0].offset
	bytesLeft := sfr.sp[0].numBytes - posInFragment
	if int64(len(b)) > bytesLeft {
		b = b[0:bytesLeft]
	}

	n, err = sfr.rfr.Read(b)
	sfr.pos += int64(n)

	if int64(n) == bytesLeft {
		// We're done with this fragment
		sfr.sp = sfr.sp[1:]
	}

	if err == io.EOF && sfr.pos < sfr.tot {
		// We reached the end of the last fragment's data, but there's a final hole
		err = nil
	}
	return
}

// numBytes returns the number of bytes left to read in the sparse file's
// sparse-encoded data in the tar archive.
func (sfr *sparseFileReader) numBytes() int64 {
	return sfr.rfr.nb
}
