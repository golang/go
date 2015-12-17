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
	"math"
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
	r       io.Reader
	err     error
	pad     int64           // amount of padding (ignored) after current file entry
	curr    numBytesReader  // reader for current file entry
	hdrBuff [blockSize]byte // buffer to use in readHeader
}

type parser struct {
	err error // Last error seen
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

// A sparseFileReader is a numBytesReader for reading sparse file data from a
// tar archive.
type sparseFileReader struct {
	rfr   numBytesReader // Reads the sparse-encoded file data
	sp    []sparseEntry  // The sparse map for the file
	pos   int64          // Keeps track of file position
	total int64          // Total size of the file
}

// A sparseEntry holds a single entry in a sparse file's sparse map.
//
// Sparse files are represented using a series of sparseEntrys.
// Despite the name, a sparseEntry represents an actual data fragment that
// references data found in the underlying archive stream. All regions not
// covered by a sparseEntry are logically filled with zeros.
//
// For example, if the underlying raw file contains the 10-byte data:
//	var compactData = "abcdefgh"
//
// And the sparse map has the following entries:
//	var sp = []sparseEntry{
//		{offset: 2,  numBytes: 5} // Data fragment for [2..7]
//		{offset: 18, numBytes: 3} // Data fragment for [18..21]
//	}
//
// Then the content of the resulting sparse file with a "real" size of 25 is:
//	var sparseData = "\x00"*2 + "abcde" + "\x00"*11 + "fgh" + "\x00"*4
type sparseEntry struct {
	offset   int64 // Starting position of the fragment
	numBytes int64 // Length of the fragment
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
//
// io.EOF is returned at the end of the input.
func (tr *Reader) Next() (*Header, error) {
	if tr.err != nil {
		return nil, tr.err
	}

	var hdr *Header
	var extHdrs map[string]string

	// Externally, Next iterates through the tar archive as if it is a series of
	// files. Internally, the tar format often uses fake "files" to add meta
	// data that describes the next file. These meta data "files" should not
	// normally be visible to the outside. As such, this loop iterates through
	// one or more "header files" until it finds a "normal file".
loop:
	for {
		tr.err = tr.skipUnread()
		if tr.err != nil {
			return nil, tr.err
		}

		hdr = tr.readHeader()
		if tr.err != nil {
			return nil, tr.err
		}

		// Check for PAX/GNU special headers and files.
		switch hdr.Typeflag {
		case TypeXHeader:
			extHdrs, tr.err = parsePAX(tr)
			if tr.err != nil {
				return nil, tr.err
			}
			continue loop // This is a meta header affecting the next header
		case TypeGNULongName, TypeGNULongLink:
			var realname []byte
			realname, tr.err = ioutil.ReadAll(tr)
			if tr.err != nil {
				return nil, tr.err
			}

			// Convert GNU extensions to use PAX headers.
			if extHdrs == nil {
				extHdrs = make(map[string]string)
			}
			var p parser
			switch hdr.Typeflag {
			case TypeGNULongName:
				extHdrs[paxPath] = p.parseString(realname)
			case TypeGNULongLink:
				extHdrs[paxLinkpath] = p.parseString(realname)
			}
			if p.err != nil {
				tr.err = p.err
				return nil, tr.err
			}
			continue loop // This is a meta header affecting the next header
		default:
			mergePAX(hdr, extHdrs)

			// Check for a PAX format sparse file
			sp, err := tr.checkForGNUSparsePAXHeaders(hdr, extHdrs)
			if err != nil {
				tr.err = err
				return nil, err
			}
			if sp != nil {
				// Current file is a PAX format GNU sparse file.
				// Set the current file reader to a sparse file reader.
				tr.curr, tr.err = newSparseFileReader(tr.curr, sp, hdr.Size)
				if tr.err != nil {
					return nil, tr.err
				}
			}
			break loop // This is a file, so stop
		}
	}
	return hdr, nil
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
	sbuf := string(buf)

	// For GNU PAX sparse format 0.0 support.
	// This function transforms the sparse format 0.0 headers into sparse format 0.1 headers.
	var sparseMap bytes.Buffer

	headers := make(map[string]string)
	// Each record is constructed as
	//     "%d %s=%s\n", length, keyword, value
	for len(sbuf) > 0 {
		key, value, residual, err := parsePAXRecord(sbuf)
		if err != nil {
			return nil, ErrHeader
		}
		sbuf = residual

		keyStr := string(key)
		if keyStr == paxGNUSparseOffset || keyStr == paxGNUSparseNumBytes {
			// GNU sparse format 0.0 special key. Write to sparseMap instead of using the headers map.
			sparseMap.WriteString(value)
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

// parsePAXRecord parses the input PAX record string into a key-value pair.
// If parsing is successful, it will slice off the currently read record and
// return the remainder as r.
//
// A PAX record is of the following form:
//	"%d %s=%s\n" % (size, key, value)
func parsePAXRecord(s string) (k, v, r string, err error) {
	// The size field ends at the first space.
	sp := strings.IndexByte(s, ' ')
	if sp == -1 {
		return "", "", s, ErrHeader
	}

	// Parse the first token as a decimal integer.
	n, perr := strconv.ParseInt(s[:sp], 10, 0) // Intentionally parse as native int
	if perr != nil || n < 5 || int64(len(s)) < n {
		return "", "", s, ErrHeader
	}

	// Extract everything between the space and the final newline.
	rec, nl, rem := s[sp+1:n-1], s[n-1:n], s[n:]
	if nl != "\n" {
		return "", "", s, ErrHeader
	}

	// The first equals separates the key from the value.
	eq := strings.IndexByte(rec, '=')
	if eq == -1 {
		return "", "", s, ErrHeader
	}
	return rec[:eq], rec[eq+1:], rem, nil
}

// parseString parses bytes as a NUL-terminated C-style string.
// If a NUL byte is not found then the whole slice is returned as a string.
func (*parser) parseString(b []byte) string {
	n := 0
	for n < len(b) && b[n] != 0 {
		n++
	}
	return string(b[0:n])
}

// parseNumeric parses the input as being encoded in either base-256 or octal.
// This function may return negative numbers.
// If parsing fails or an integer overflow occurs, err will be set.
func (p *parser) parseNumeric(b []byte) int64 {
	// Check for base-256 (binary) format first.
	// If the first bit is set, then all following bits constitute a two's
	// complement encoded number in big-endian byte order.
	if len(b) > 0 && b[0]&0x80 != 0 {
		// Handling negative numbers relies on the following identity:
		//	-a-1 == ^a
		//
		// If the number is negative, we use an inversion mask to invert the
		// data bytes and treat the value as an unsigned number.
		var inv byte // 0x00 if positive or zero, 0xff if negative
		if b[0]&0x40 != 0 {
			inv = 0xff
		}

		var x uint64
		for i, c := range b {
			c ^= inv // Inverts c only if inv is 0xff, otherwise does nothing
			if i == 0 {
				c &= 0x7f // Ignore signal bit in first byte
			}
			if (x >> 56) > 0 {
				p.err = ErrHeader // Integer overflow
				return 0
			}
			x = x<<8 | uint64(c)
		}
		if (x >> 63) > 0 {
			p.err = ErrHeader // Integer overflow
			return 0
		}
		if inv == 0xff {
			return ^int64(x)
		}
		return int64(x)
	}

	// Normal case is base-8 (octal) format.
	return p.parseOctal(b)
}

func (p *parser) parseOctal(b []byte) int64 {
	// Because unused fields are filled with NULs, we need
	// to skip leading NULs. Fields may also be padded with
	// spaces or NULs.
	// So we remove leading and trailing NULs and spaces to
	// be sure.
	b = bytes.Trim(b, " \x00")

	if len(b) == 0 {
		return 0
	}
	x, perr := strconv.ParseUint(p.parseString(b), 8, 64)
	if perr != nil {
		p.err = ErrHeader
	}
	return int64(x)
}

// skipUnread skips any unread bytes in the existing file entry, as well as any
// alignment padding. It returns io.ErrUnexpectedEOF if any io.EOF is
// encountered in the data portion; it is okay to hit io.EOF in the padding.
//
// Note that this function still works properly even when sparse files are being
// used since numBytes returns the bytes remaining in the underlying io.Reader.
func (tr *Reader) skipUnread() error {
	dataSkip := tr.numBytes()      // Number of data bytes to skip
	totalSkip := dataSkip + tr.pad // Total number of bytes to skip
	tr.curr, tr.pad = nil, 0

	// If possible, Seek to the last byte before the end of the data section.
	// Do this because Seek is often lazy about reporting errors; this will mask
	// the fact that the tar stream may be truncated. We can rely on the
	// io.CopyN done shortly afterwards to trigger any IO errors.
	var seekSkipped int64 // Number of bytes skipped via Seek
	if sr, ok := tr.r.(io.Seeker); ok && dataSkip > 1 {
		// Not all io.Seeker can actually Seek. For example, os.Stdin implements
		// io.Seeker, but calling Seek always returns an error and performs
		// no action. Thus, we try an innocent seek to the current position
		// to see if Seek is really supported.
		pos1, err := sr.Seek(0, os.SEEK_CUR)
		if err == nil {
			// Seek seems supported, so perform the real Seek.
			pos2, err := sr.Seek(dataSkip-1, os.SEEK_CUR)
			if err != nil {
				tr.err = err
				return tr.err
			}
			seekSkipped = pos2 - pos1
		}
	}

	var copySkipped int64 // Number of bytes skipped via CopyN
	copySkipped, tr.err = io.CopyN(ioutil.Discard, tr.r, totalSkip-seekSkipped)
	if tr.err == io.EOF && seekSkipped+copySkipped < dataSkip {
		tr.err = io.ErrUnexpectedEOF
	}
	return tr.err
}

func (tr *Reader) verifyChecksum(header []byte) bool {
	if tr.err != nil {
		return false
	}

	var p parser
	given := p.parseOctal(header[148:156])
	unsigned, signed := checksum(header)
	return p.err == nil && (given == unsigned || given == signed)
}

// readHeader reads the next block header and assumes that the underlying reader
// is already aligned to a block boundary.
//
// The err will be set to io.EOF only when one of the following occurs:
//	* Exactly 0 bytes are read and EOF is hit.
//	* Exactly 1 block of zeros is read and EOF is hit.
//	* At least 2 blocks of zeros are read.
func (tr *Reader) readHeader() *Header {
	header := tr.hdrBuff[:]
	copy(header, zeroBlock)

	if _, tr.err = io.ReadFull(tr.r, header); tr.err != nil {
		return nil // io.EOF is okay here
	}

	// Two blocks of zero bytes marks the end of the archive.
	if bytes.Equal(header, zeroBlock[0:blockSize]) {
		if _, tr.err = io.ReadFull(tr.r, header); tr.err != nil {
			return nil // io.EOF is okay here
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
	var p parser
	hdr := new(Header)
	s := slicer(header)

	hdr.Name = p.parseString(s.next(100))
	hdr.Mode = p.parseNumeric(s.next(8))
	hdr.Uid = int(p.parseNumeric(s.next(8)))
	hdr.Gid = int(p.parseNumeric(s.next(8)))
	hdr.Size = p.parseNumeric(s.next(12))
	hdr.ModTime = time.Unix(p.parseNumeric(s.next(12)), 0)
	s.next(8) // chksum
	hdr.Typeflag = s.next(1)[0]
	hdr.Linkname = p.parseString(s.next(100))

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
		hdr.Uname = p.parseString(s.next(32))
		hdr.Gname = p.parseString(s.next(32))
		devmajor := s.next(8)
		devminor := s.next(8)
		if hdr.Typeflag == TypeChar || hdr.Typeflag == TypeBlock {
			hdr.Devmajor = p.parseNumeric(devmajor)
			hdr.Devminor = p.parseNumeric(devminor)
		}
		var prefix string
		switch format {
		case "posix", "gnu":
			prefix = p.parseString(s.next(155))
		case "star":
			prefix = p.parseString(s.next(131))
			hdr.AccessTime = time.Unix(p.parseNumeric(s.next(12)), 0)
			hdr.ChangeTime = time.Unix(p.parseNumeric(s.next(12)), 0)
		}
		if len(prefix) > 0 {
			hdr.Name = prefix + "/" + hdr.Name
		}
	}

	if p.err != nil {
		tr.err = p.err
		return nil
	}

	nb := hdr.Size
	if isHeaderOnlyType(hdr.Typeflag) {
		nb = 0
	}
	if nb < 0 {
		tr.err = ErrHeader
		return nil
	}

	// Set the current file reader.
	tr.pad = -nb & (blockSize - 1) // blockSize is a power of two
	tr.curr = &regFileReader{r: tr.r, nb: nb}

	// Check for old GNU sparse format entry.
	if hdr.Typeflag == TypeGNUSparse {
		// Get the real size of the file.
		hdr.Size = p.parseNumeric(header[483:495])
		if p.err != nil {
			tr.err = p.err
			return nil
		}

		// Read the sparse map.
		sp := tr.readOldGNUSparseMap(header)
		if tr.err != nil {
			return nil
		}

		// Current file is a GNU sparse file. Update the current file reader.
		tr.curr, tr.err = newSparseFileReader(tr.curr, sp, hdr.Size)
		if tr.err != nil {
			return nil
		}
	}

	return hdr
}

// readOldGNUSparseMap reads the sparse map as stored in the old GNU sparse format.
// The sparse map is stored in the tar header if it's small enough. If it's larger than four entries,
// then one or more extension headers are used to store the rest of the sparse map.
func (tr *Reader) readOldGNUSparseMap(header []byte) []sparseEntry {
	var p parser
	isExtended := header[oldGNUSparseMainHeaderIsExtendedOffset] != 0
	spCap := oldGNUSparseMainHeaderNumEntries
	if isExtended {
		spCap += oldGNUSparseExtendedHeaderNumEntries
	}
	sp := make([]sparseEntry, 0, spCap)
	s := slicer(header[oldGNUSparseMainHeaderOffset:])

	// Read the four entries from the main tar header
	for i := 0; i < oldGNUSparseMainHeaderNumEntries; i++ {
		offset := p.parseNumeric(s.next(oldGNUSparseOffsetSize))
		numBytes := p.parseNumeric(s.next(oldGNUSparseNumBytesSize))
		if p.err != nil {
			tr.err = p.err
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
			offset := p.parseNumeric(s.next(oldGNUSparseOffsetSize))
			numBytes := p.parseNumeric(s.next(oldGNUSparseNumBytesSize))
			if p.err != nil {
				tr.err = p.err
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

// readGNUSparseMap1x0 reads the sparse map as stored in GNU's PAX sparse format
// version 1.0. The format of the sparse map consists of a series of
// newline-terminated numeric fields. The first field is the number of entries
// and is always present. Following this are the entries, consisting of two
// fields (offset, numBytes). This function must stop reading at the end
// boundary of the block containing the last newline.
//
// Note that the GNU manual says that numeric values should be encoded in octal
// format. However, the GNU tar utility itself outputs these values in decimal.
// As such, this library treats values as being encoded in decimal.
func readGNUSparseMap1x0(r io.Reader) ([]sparseEntry, error) {
	var cntNewline int64
	var buf bytes.Buffer
	var blk = make([]byte, blockSize)

	// feedTokens copies data in numBlock chunks from r into buf until there are
	// at least cnt newlines in buf. It will not read more blocks than needed.
	var feedTokens = func(cnt int64) error {
		for cntNewline < cnt {
			if _, err := io.ReadFull(r, blk); err != nil {
				if err == io.EOF {
					err = io.ErrUnexpectedEOF
				}
				return err
			}
			buf.Write(blk)
			for _, c := range blk {
				if c == '\n' {
					cntNewline++
				}
			}
		}
		return nil
	}

	// nextToken gets the next token delimited by a newline. This assumes that
	// at least one newline exists in the buffer.
	var nextToken = func() string {
		cntNewline--
		tok, _ := buf.ReadString('\n')
		return tok[:len(tok)-1] // Cut off newline
	}

	// Parse for the number of entries.
	// Use integer overflow resistant math to check this.
	if err := feedTokens(1); err != nil {
		return nil, err
	}
	numEntries, err := strconv.ParseInt(nextToken(), 10, 0) // Intentionally parse as native int
	if err != nil || numEntries < 0 || int(2*numEntries) < int(numEntries) {
		return nil, ErrHeader
	}

	// Parse for all member entries.
	// numEntries is trusted after this since a potential attacker must have
	// committed resources proportional to what this library used.
	if err := feedTokens(2 * numEntries); err != nil {
		return nil, err
	}
	sp := make([]sparseEntry, 0, numEntries)
	for i := int64(0); i < numEntries; i++ {
		offset, err := strconv.ParseInt(nextToken(), 10, 64)
		if err != nil {
			return nil, ErrHeader
		}
		numBytes, err := strconv.ParseInt(nextToken(), 10, 64)
		if err != nil {
			return nil, ErrHeader
		}
		sp = append(sp, sparseEntry{offset: offset, numBytes: numBytes})
	}
	return sp, nil
}

// readGNUSparseMap0x1 reads the sparse map as stored in GNU's PAX sparse format
// version 0.1. The sparse map is stored in the PAX headers.
func readGNUSparseMap0x1(extHdrs map[string]string) ([]sparseEntry, error) {
	// Get number of entries.
	// Use integer overflow resistant math to check this.
	numEntriesStr := extHdrs[paxGNUSparseNumBlocks]
	numEntries, err := strconv.ParseInt(numEntriesStr, 10, 0) // Intentionally parse as native int
	if err != nil || numEntries < 0 || int(2*numEntries) < int(numEntries) {
		return nil, ErrHeader
	}

	// There should be two numbers in sparseMap for each entry.
	sparseMap := strings.Split(extHdrs[paxGNUSparseMap], ",")
	if int64(len(sparseMap)) != 2*numEntries {
		return nil, ErrHeader
	}

	// Loop through the entries in the sparse map.
	// numEntries is trusted now.
	sp := make([]sparseEntry, 0, numEntries)
	for i := int64(0); i < numEntries; i++ {
		offset, err := strconv.ParseInt(sparseMap[2*i], 10, 64)
		if err != nil {
			return nil, ErrHeader
		}
		numBytes, err := strconv.ParseInt(sparseMap[2*i+1], 10, 64)
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
//
// Calling Read on special types like TypeLink, TypeSymLink, TypeChar,
// TypeBlock, TypeDir, and TypeFifo returns 0, io.EOF regardless of what
// the Header.Size claims.
func (tr *Reader) Read(b []byte) (n int, err error) {
	if tr.err != nil {
		return 0, tr.err
	}
	if tr.curr == nil {
		return 0, io.EOF
	}

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

// newSparseFileReader creates a new sparseFileReader, but validates all of the
// sparse entries before doing so.
func newSparseFileReader(rfr numBytesReader, sp []sparseEntry, total int64) (*sparseFileReader, error) {
	if total < 0 {
		return nil, ErrHeader // Total size cannot be negative
	}

	// Validate all sparse entries. These are the same checks as performed by
	// the BSD tar utility.
	for i, s := range sp {
		switch {
		case s.offset < 0 || s.numBytes < 0:
			return nil, ErrHeader // Negative values are never okay
		case s.offset > math.MaxInt64-s.numBytes:
			return nil, ErrHeader // Integer overflow with large length
		case s.offset+s.numBytes > total:
			return nil, ErrHeader // Region extends beyond the "real" size
		case i > 0 && sp[i-1].offset+sp[i-1].numBytes > s.offset:
			return nil, ErrHeader // Regions can't overlap and must be in order
		}
	}
	return &sparseFileReader{rfr: rfr, sp: sp, total: total}, nil
}

// readHole reads a sparse hole ending at endOffset.
func (sfr *sparseFileReader) readHole(b []byte, endOffset int64) int {
	n64 := endOffset - sfr.pos
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
	// Skip past all empty fragments.
	for len(sfr.sp) > 0 && sfr.sp[0].numBytes == 0 {
		sfr.sp = sfr.sp[1:]
	}

	// If there are no more fragments, then it is possible that there
	// is one last sparse hole.
	if len(sfr.sp) == 0 {
		// This behavior matches the BSD tar utility.
		// However, GNU tar stops returning data even if sfr.total is unmet.
		if sfr.pos < sfr.total {
			return sfr.readHole(b, sfr.total), nil
		}
		return 0, io.EOF
	}

	// In front of a data fragment, so read a hole.
	if sfr.pos < sfr.sp[0].offset {
		return sfr.readHole(b, sfr.sp[0].offset), nil
	}

	// In a data fragment, so read from it.
	// This math is overflow free since we verify that offset and numBytes can
	// be safely added when creating the sparseFileReader.
	endPos := sfr.sp[0].offset + sfr.sp[0].numBytes // End offset of fragment
	bytesLeft := endPos - sfr.pos                   // Bytes left in fragment
	if int64(len(b)) > bytesLeft {
		b = b[:bytesLeft]
	}

	n, err = sfr.rfr.Read(b)
	sfr.pos += int64(n)
	if err == io.EOF {
		if sfr.pos < endPos {
			err = io.ErrUnexpectedEOF // There was supposed to be more data
		} else if sfr.pos < sfr.total {
			err = nil // There is still an implicit sparse hole at the end
		}
	}

	if sfr.pos == endPos {
		sfr.sp = sfr.sp[1:] // We are done with this fragment, so pop it
	}
	return n, err
}

// numBytes returns the number of bytes left to read in the sparse file's
// sparse-encoded data in the tar archive.
func (sfr *sparseFileReader) numBytes() int64 {
	return sfr.rfr.numBytes()
}
