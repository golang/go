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
	"strconv"
	"strings"
	"time"
)

var (
	ErrHeader = errors.New("archive/tar: invalid tar header")
)

// A Reader provides sequential access to the contents of a tar archive.
// A tar archive consists of a sequence of files.
// The Next method advances to the next file in the archive (including the first),
// and then it can be treated as an io.Reader to access the file's data.
type Reader struct {
	r    io.Reader
	pad  int64          // amount of padding (ignored) after current file entry
	curr numBytesReader // reader for current file entry
	blk  block          // buffer to use as temporary local storage

	// err is a persistent error.
	// It is only the responsibility of every exported method of Reader to
	// ensure that this error is sticky.
	err error
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

// NewReader creates a new Reader reading from r.
func NewReader(r io.Reader) *Reader { return &Reader{r: r} }

// Next advances to the next entry in the tar archive.
//
// io.EOF is returned at the end of the input.
func (tr *Reader) Next() (*Header, error) {
	if tr.err != nil {
		return nil, tr.err
	}
	hdr, err := tr.next()
	tr.err = err
	return hdr, err
}

func (tr *Reader) next() (*Header, error) {
	var extHdrs map[string]string

	// Externally, Next iterates through the tar archive as if it is a series of
	// files. Internally, the tar format often uses fake "files" to add meta
	// data that describes the next file. These meta data "files" should not
	// normally be visible to the outside. As such, this loop iterates through
	// one or more "header files" until it finds a "normal file".
loop:
	for {
		if err := tr.skipUnread(); err != nil {
			return nil, err
		}
		hdr, rawHdr, err := tr.readHeader()
		if err != nil {
			return nil, err
		}
		if err := tr.handleRegularFile(hdr); err != nil {
			return nil, err
		}

		// Check for PAX/GNU special headers and files.
		switch hdr.Typeflag {
		case TypeXHeader:
			extHdrs, err = parsePAX(tr)
			if err != nil {
				return nil, err
			}
			continue loop // This is a meta header affecting the next header
		case TypeGNULongName, TypeGNULongLink:
			realname, err := ioutil.ReadAll(tr)
			if err != nil {
				return nil, err
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
				return nil, p.err
			}
			continue loop // This is a meta header affecting the next header
		default:
			// The old GNU sparse format is handled here since it is technically
			// just a regular file with additional attributes.

			if err := mergePAX(hdr, extHdrs); err != nil {
				return nil, err
			}

			// The extended headers may have updated the size.
			// Thus, setup the regFileReader again after merging PAX headers.
			if err := tr.handleRegularFile(hdr); err != nil {
				return nil, err
			}

			// Sparse formats rely on being able to read from the logical data
			// section; there must be a preceding call to handleRegularFile.
			if err := tr.handleSparseFile(hdr, rawHdr, extHdrs); err != nil {
				return nil, err
			}
			return hdr, nil // This is a file, so stop
		}
	}
}

// handleRegularFile sets up the current file reader and padding such that it
// can only read the following logical data section. It will properly handle
// special headers that contain no data section.
func (tr *Reader) handleRegularFile(hdr *Header) error {
	nb := hdr.Size
	if isHeaderOnlyType(hdr.Typeflag) {
		nb = 0
	}
	if nb < 0 {
		return ErrHeader
	}

	tr.pad = -nb & (blockSize - 1) // blockSize is a power of two
	tr.curr = &regFileReader{r: tr.r, nb: nb}
	return nil
}

// handleSparseFile checks if the current file is a sparse format of any type
// and sets the curr reader appropriately.
func (tr *Reader) handleSparseFile(hdr *Header, rawHdr *block, extHdrs map[string]string) error {
	var sp []sparseEntry
	var err error
	if hdr.Typeflag == TypeGNUSparse {
		sp, err = tr.readOldGNUSparseMap(hdr, rawHdr)
		if err != nil {
			return err
		}
	} else {
		sp, err = tr.checkForGNUSparsePAXHeaders(hdr, extHdrs)
		if err != nil {
			return err
		}
	}

	// If sp is non-nil, then this is a sparse file.
	// Note that it is possible for len(sp) to be zero.
	if sp != nil {
		tr.curr, err = newSparseFileReader(tr.curr, sp, hdr.Size)
	}
	return err
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
		realSize, err := strconv.ParseInt(sparseSize, 10, 64)
		if err != nil {
			return nil, ErrHeader
		}
		hdr.Size = realSize
	} else if sparseRealSizeOk {
		realSize, err := strconv.ParseInt(sparseRealSize, 10, 64)
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
func mergePAX(hdr *Header, headers map[string]string) (err error) {
	var id64 int64
	for k, v := range headers {
		switch k {
		case paxPath:
			hdr.Name = v
		case paxLinkpath:
			hdr.Linkname = v
		case paxUname:
			hdr.Uname = v
		case paxGname:
			hdr.Gname = v
		case paxUid:
			id64, err = strconv.ParseInt(v, 10, 64)
			hdr.Uid = int(id64) // Integer overflow possible
		case paxGid:
			id64, err = strconv.ParseInt(v, 10, 64)
			hdr.Gid = int(id64) // Integer overflow possible
		case paxAtime:
			hdr.AccessTime, err = parsePAXTime(v)
		case paxMtime:
			hdr.ModTime, err = parsePAXTime(v)
		case paxCtime:
			hdr.ChangeTime, err = parsePAXTime(v)
		case paxSize:
			hdr.Size, err = strconv.ParseInt(v, 10, 64)
		default:
			if strings.HasPrefix(k, paxXattr) {
				if hdr.Xattrs == nil {
					hdr.Xattrs = make(map[string]string)
				}
				hdr.Xattrs[k[len(paxXattr):]] = v
			}
		}
		if err != nil {
			return ErrHeader
		}
	}
	return nil
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
	// This function transforms the sparse format 0.0 headers into format 0.1
	// headers since 0.0 headers were not PAX compliant.
	var sparseMap []string

	extHdrs := make(map[string]string)
	for len(sbuf) > 0 {
		key, value, residual, err := parsePAXRecord(sbuf)
		if err != nil {
			return nil, ErrHeader
		}
		sbuf = residual

		switch key {
		case paxGNUSparseOffset, paxGNUSparseNumBytes:
			// Validate sparse header order and value.
			if (len(sparseMap)%2 == 0 && key != paxGNUSparseOffset) ||
				(len(sparseMap)%2 == 1 && key != paxGNUSparseNumBytes) ||
				strings.Contains(value, ",") {
				return nil, ErrHeader
			}
			sparseMap = append(sparseMap, value)
		default:
			// According to PAX specification, a value is stored only if it is
			// non-empty. Otherwise, the key is deleted.
			if len(value) > 0 {
				extHdrs[key] = value
			} else {
				delete(extHdrs, key)
			}
		}
	}
	if len(sparseMap) > 0 {
		extHdrs[paxGNUSparseMap] = strings.Join(sparseMap, ",")
	}
	return extHdrs, nil
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
		pos1, err := sr.Seek(0, io.SeekCurrent)
		if err == nil {
			// Seek seems supported, so perform the real Seek.
			pos2, err := sr.Seek(dataSkip-1, io.SeekCurrent)
			if err != nil {
				return err
			}
			seekSkipped = pos2 - pos1
		}
	}

	copySkipped, err := io.CopyN(ioutil.Discard, tr.r, totalSkip-seekSkipped)
	if err == io.EOF && seekSkipped+copySkipped < dataSkip {
		err = io.ErrUnexpectedEOF
	}
	return err
}

// readHeader reads the next block header and assumes that the underlying reader
// is already aligned to a block boundary. It returns the raw block of the
// header in case further processing is required.
//
// The err will be set to io.EOF only when one of the following occurs:
//	* Exactly 0 bytes are read and EOF is hit.
//	* Exactly 1 block of zeros is read and EOF is hit.
//	* At least 2 blocks of zeros are read.
func (tr *Reader) readHeader() (*Header, *block, error) {
	// Two blocks of zero bytes marks the end of the archive.
	if _, err := io.ReadFull(tr.r, tr.blk[:]); err != nil {
		return nil, nil, err // EOF is okay here; exactly 0 bytes read
	}
	if bytes.Equal(tr.blk[:], zeroBlock[:]) {
		if _, err := io.ReadFull(tr.r, tr.blk[:]); err != nil {
			return nil, nil, err // EOF is okay here; exactly 1 block of zeros read
		}
		if bytes.Equal(tr.blk[:], zeroBlock[:]) {
			return nil, nil, io.EOF // normal EOF; exactly 2 block of zeros read
		}
		return nil, nil, ErrHeader // Zero block and then non-zero block
	}

	// Verify the header matches a known format.
	format := tr.blk.GetFormat()
	if format == formatUnknown {
		return nil, nil, ErrHeader
	}

	var p parser
	hdr := new(Header)

	// Unpack the V7 header.
	v7 := tr.blk.V7()
	hdr.Name = p.parseString(v7.Name())
	hdr.Mode = p.parseNumeric(v7.Mode())
	hdr.Uid = int(p.parseNumeric(v7.UID()))
	hdr.Gid = int(p.parseNumeric(v7.GID()))
	hdr.Size = p.parseNumeric(v7.Size())
	hdr.ModTime = time.Unix(p.parseNumeric(v7.ModTime()), 0)
	hdr.Typeflag = v7.TypeFlag()[0]
	hdr.Linkname = p.parseString(v7.LinkName())

	// The atime and ctime fields are often left unused. Some versions of Go
	// had a bug in the tar.Writer where it would output an invalid tar file
	// in certain rare situations because the logic incorrectly believed that
	// the old GNU format had a prefix field. This is wrong and leads to
	// an outputted file that actually mangles the atime and ctime fields.
	//
	// In order to continue reading tar files created by a buggy writer, we
	// try to parse the atime and ctime fields, but just return the zero value
	// of time.Time when we cannot parse them.
	//
	// See https://golang.org/issues/12594
	tryParseTime := func(b []byte) time.Time {
		var p parser
		n := p.parseNumeric(b)
		if b[0] != 0x00 && p.err == nil {
			return time.Unix(n, 0)
		}
		return time.Time{}
	}

	// Unpack format specific fields.
	if format > formatV7 {
		ustar := tr.blk.USTAR()
		hdr.Uname = p.parseString(ustar.UserName())
		hdr.Gname = p.parseString(ustar.GroupName())
		if hdr.Typeflag == TypeChar || hdr.Typeflag == TypeBlock {
			hdr.Devmajor = p.parseNumeric(ustar.DevMajor())
			hdr.Devminor = p.parseNumeric(ustar.DevMinor())
		}

		var prefix string
		switch format {
		case formatUSTAR:
			ustar := tr.blk.USTAR()
			prefix = p.parseString(ustar.Prefix())
		case formatSTAR:
			star := tr.blk.STAR()
			prefix = p.parseString(star.Prefix())
			hdr.AccessTime = time.Unix(p.parseNumeric(star.AccessTime()), 0)
			hdr.ChangeTime = time.Unix(p.parseNumeric(star.ChangeTime()), 0)
		case formatGNU:
			gnu := tr.blk.GNU()
			hdr.AccessTime = tryParseTime(gnu.AccessTime())
			hdr.ChangeTime = tryParseTime(gnu.ChangeTime())
		}
		if len(prefix) > 0 {
			hdr.Name = prefix + "/" + hdr.Name
		}
	}
	return hdr, &tr.blk, p.err
}

// readOldGNUSparseMap reads the sparse map from the old GNU sparse format.
// The sparse map is stored in the tar header if it's small enough.
// If it's larger than four entries, then one or more extension headers are used
// to store the rest of the sparse map.
//
// The Header.Size does not reflect the size of any extended headers used.
// Thus, this function will read from the raw io.Reader to fetch extra headers.
// This method mutates blk in the process.
func (tr *Reader) readOldGNUSparseMap(hdr *Header, blk *block) ([]sparseEntry, error) {
	// Make sure that the input format is GNU.
	// Unfortunately, the STAR format also has a sparse header format that uses
	// the same type flag but has a completely different layout.
	if blk.GetFormat() != formatGNU {
		return nil, ErrHeader
	}

	var p parser
	hdr.Size = p.parseNumeric(blk.GNU().RealSize())
	if p.err != nil {
		return nil, p.err
	}
	var s sparseArray = blk.GNU().Sparse()
	var sp = make([]sparseEntry, 0, s.MaxEntries())
	for {
		for i := 0; i < s.MaxEntries(); i++ {
			// This termination condition is identical to GNU and BSD tar.
			if s.Entry(i).Offset()[0] == 0x00 {
				break // Don't return, need to process extended headers (even if empty)
			}
			offset := p.parseNumeric(s.Entry(i).Offset())
			numBytes := p.parseNumeric(s.Entry(i).NumBytes())
			if p.err != nil {
				return nil, p.err
			}
			sp = append(sp, sparseEntry{offset: offset, numBytes: numBytes})
		}

		if s.IsExtended()[0] > 0 {
			// There are more entries. Read an extension header and parse its entries.
			if _, err := io.ReadFull(tr.r, blk[:]); err != nil {
				if err == io.EOF {
					err = io.ErrUnexpectedEOF
				}
				return nil, err
			}
			s = blk.Sparse()
			continue
		}
		return sp, nil // Done
	}
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
func (tr *Reader) Read(b []byte) (int, error) {
	if tr.err != nil {
		return 0, tr.err
	}
	if tr.curr == nil {
		return 0, io.EOF
	}

	n, err := tr.curr.Read(b)
	if err != nil && err != io.EOF {
		tr.err = err
	}
	return n, err
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
