// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

import (
	"bytes"
	"io"
	"strconv"
	"strings"
	"time"
)

// Reader provides sequential access to the contents of a tar archive.
// Reader.Next advances to the next file in the archive (including the first),
// and then Reader can be treated as an io.Reader to access the file's data.
type Reader struct {
	r    io.Reader
	pad  int64      // Amount of padding (ignored) after current file entry
	curr fileReader // Reader for current file entry
	blk  block      // Buffer to use as temporary local storage

	// err is a persistent error.
	// It is only the responsibility of every exported method of Reader to
	// ensure that this error is sticky.
	err error
}

type fileReader interface {
	io.Reader
	fileState

	WriteTo(io.Writer) (int64, error)
}

// NewReader creates a new Reader reading from r.
func NewReader(r io.Reader) *Reader {
	return &Reader{r: r, curr: &regFileReader{r, 0}}
}

// Next advances to the next entry in the tar archive.
// The Header.Size determines how many bytes can be read for the next file.
// Any remaining data in the current file is automatically discarded.
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
	var paxHdrs map[string]string
	var gnuLongName, gnuLongLink string

	// Externally, Next iterates through the tar archive as if it is a series of
	// files. Internally, the tar format often uses fake "files" to add meta
	// data that describes the next file. These meta data "files" should not
	// normally be visible to the outside. As such, this loop iterates through
	// one or more "header files" until it finds a "normal file".
	format := FormatUSTAR | FormatPAX | FormatGNU
	for {
		// Discard the remainder of the file and any padding.
		if err := discard(tr.r, tr.curr.physicalRemaining()); err != nil {
			return nil, err
		}
		if _, err := tryReadFull(tr.r, tr.blk[:tr.pad]); err != nil {
			return nil, err
		}
		tr.pad = 0

		hdr, rawHdr, err := tr.readHeader()
		if err != nil {
			return nil, err
		}
		if err := tr.handleRegularFile(hdr); err != nil {
			return nil, err
		}
		format.mayOnlyBe(hdr.Format)

		// Check for PAX/GNU special headers and files.
		switch hdr.Typeflag {
		case TypeXHeader, TypeXGlobalHeader:
			format.mayOnlyBe(FormatPAX)
			paxHdrs, err = parsePAX(tr)
			if err != nil {
				return nil, err
			}
			if hdr.Typeflag == TypeXGlobalHeader {
				mergePAX(hdr, paxHdrs)
				return &Header{
					Name:       hdr.Name,
					Typeflag:   hdr.Typeflag,
					Xattrs:     hdr.Xattrs,
					PAXRecords: hdr.PAXRecords,
					Format:     format,
				}, nil
			}
			continue // This is a meta header affecting the next header
		case TypeGNULongName, TypeGNULongLink:
			format.mayOnlyBe(FormatGNU)
			realname, err := readSpecialFile(tr)
			if err != nil {
				return nil, err
			}

			var p parser
			switch hdr.Typeflag {
			case TypeGNULongName:
				gnuLongName = p.parseString(realname)
			case TypeGNULongLink:
				gnuLongLink = p.parseString(realname)
			}
			continue // This is a meta header affecting the next header
		default:
			// The old GNU sparse format is handled here since it is technically
			// just a regular file with additional attributes.

			if err := mergePAX(hdr, paxHdrs); err != nil {
				return nil, err
			}
			if gnuLongName != "" {
				hdr.Name = gnuLongName
			}
			if gnuLongLink != "" {
				hdr.Linkname = gnuLongLink
			}
			if hdr.Typeflag == TypeRegA {
				if strings.HasSuffix(hdr.Name, "/") {
					hdr.Typeflag = TypeDir // Legacy archives use trailing slash for directories
				} else {
					hdr.Typeflag = TypeReg
				}
			}

			// The extended headers may have updated the size.
			// Thus, setup the regFileReader again after merging PAX headers.
			if err := tr.handleRegularFile(hdr); err != nil {
				return nil, err
			}

			// Sparse formats rely on being able to read from the logical data
			// section; there must be a preceding call to handleRegularFile.
			if err := tr.handleSparseFile(hdr, rawHdr); err != nil {
				return nil, err
			}

			// Set the final guess at the format.
			if format.has(FormatUSTAR) && format.has(FormatPAX) {
				format.mayOnlyBe(FormatUSTAR)
			}
			hdr.Format = format
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

	tr.pad = blockPadding(nb)
	tr.curr = &regFileReader{r: tr.r, nb: nb}
	return nil
}

// handleSparseFile checks if the current file is a sparse format of any type
// and sets the curr reader appropriately.
func (tr *Reader) handleSparseFile(hdr *Header, rawHdr *block) error {
	var spd sparseDatas
	var err error
	if hdr.Typeflag == TypeGNUSparse {
		spd, err = tr.readOldGNUSparseMap(hdr, rawHdr)
	} else {
		spd, err = tr.readGNUSparsePAXHeaders(hdr)
	}

	// If sp is non-nil, then this is a sparse file.
	// Note that it is possible for len(sp) == 0.
	if err == nil && spd != nil {
		if isHeaderOnlyType(hdr.Typeflag) || !validateSparseEntries(spd, hdr.Size) {
			return ErrHeader
		}
		sph := invertSparseEntries(spd, hdr.Size)
		tr.curr = &sparseFileReader{tr.curr, sph, 0}
	}
	return err
}

// readGNUSparsePAXHeaders checks the PAX headers for GNU sparse headers.
// If they are found, then this function reads the sparse map and returns it.
// This assumes that 0.0 headers have already been converted to 0.1 headers
// by the PAX header parsing logic.
func (tr *Reader) readGNUSparsePAXHeaders(hdr *Header) (sparseDatas, error) {
	// Identify the version of GNU headers.
	var is1x0 bool
	major, minor := hdr.PAXRecords[paxGNUSparseMajor], hdr.PAXRecords[paxGNUSparseMinor]
	switch {
	case major == "0" && (minor == "0" || minor == "1"):
		is1x0 = false
	case major == "1" && minor == "0":
		is1x0 = true
	case major != "" || minor != "":
		return nil, nil // Unknown GNU sparse PAX version
	case hdr.PAXRecords[paxGNUSparseMap] != "":
		is1x0 = false // 0.0 and 0.1 did not have explicit version records, so guess
	default:
		return nil, nil // Not a PAX format GNU sparse file.
	}
	hdr.Format.mayOnlyBe(FormatPAX)

	// Update hdr from GNU sparse PAX headers.
	if name := hdr.PAXRecords[paxGNUSparseName]; name != "" {
		hdr.Name = name
	}
	size := hdr.PAXRecords[paxGNUSparseSize]
	if size == "" {
		size = hdr.PAXRecords[paxGNUSparseRealSize]
	}
	if size != "" {
		n, err := strconv.ParseInt(size, 10, 64)
		if err != nil {
			return nil, ErrHeader
		}
		hdr.Size = n
	}

	// Read the sparse map according to the appropriate format.
	if is1x0 {
		return readGNUSparseMap1x0(tr.curr)
	}
	return readGNUSparseMap0x1(hdr.PAXRecords)
}

// mergePAX merges paxHdrs into hdr for all relevant fields of Header.
func mergePAX(hdr *Header, paxHdrs map[string]string) (err error) {
	for k, v := range paxHdrs {
		if v == "" {
			continue // Keep the original USTAR value
		}
		var id64 int64
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
			if strings.HasPrefix(k, paxSchilyXattr) {
				if hdr.Xattrs == nil {
					hdr.Xattrs = make(map[string]string)
				}
				hdr.Xattrs[k[len(paxSchilyXattr):]] = v
			}
		}
		if err != nil {
			return ErrHeader
		}
	}
	hdr.PAXRecords = paxHdrs
	return nil
}

// parsePAX parses PAX headers.
// If an extended header (type 'x') is invalid, ErrHeader is returned
func parsePAX(r io.Reader) (map[string]string, error) {
	buf, err := readSpecialFile(r)
	if err != nil {
		return nil, err
	}
	sbuf := string(buf)

	// For GNU PAX sparse format 0.0 support.
	// This function transforms the sparse format 0.0 headers into format 0.1
	// headers since 0.0 headers were not PAX compliant.
	var sparseMap []string

	paxHdrs := make(map[string]string)
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
			paxHdrs[key] = value
		}
	}
	if len(sparseMap) > 0 {
		paxHdrs[paxGNUSparseMap] = strings.Join(sparseMap, ",")
	}
	return paxHdrs, nil
}

// readHeader reads the next block header and assumes that the underlying reader
// is already aligned to a block boundary. It returns the raw block of the
// header in case further processing is required.
//
// The err will be set to io.EOF only when one of the following occurs:
//   - Exactly 0 bytes are read and EOF is hit.
//   - Exactly 1 block of zeros is read and EOF is hit.
//   - At least 2 blocks of zeros are read.
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
	format := tr.blk.getFormat()
	if format == FormatUnknown {
		return nil, nil, ErrHeader
	}

	var p parser
	hdr := new(Header)

	// Unpack the V7 header.
	v7 := tr.blk.toV7()
	hdr.Typeflag = v7.typeFlag()[0]
	hdr.Name = p.parseString(v7.name())
	hdr.Linkname = p.parseString(v7.linkName())
	hdr.Size = p.parseNumeric(v7.size())
	hdr.Mode = p.parseNumeric(v7.mode())
	hdr.Uid = int(p.parseNumeric(v7.uid()))
	hdr.Gid = int(p.parseNumeric(v7.gid()))
	hdr.ModTime = time.Unix(p.parseNumeric(v7.modTime()), 0)

	// Unpack format specific fields.
	if format > formatV7 {
		ustar := tr.blk.toUSTAR()
		hdr.Uname = p.parseString(ustar.userName())
		hdr.Gname = p.parseString(ustar.groupName())
		hdr.Devmajor = p.parseNumeric(ustar.devMajor())
		hdr.Devminor = p.parseNumeric(ustar.devMinor())

		var prefix string
		switch {
		case format.has(FormatUSTAR | FormatPAX):
			hdr.Format = format
			ustar := tr.blk.toUSTAR()
			prefix = p.parseString(ustar.prefix())

			// For Format detection, check if block is properly formatted since
			// the parser is more liberal than what USTAR actually permits.
			notASCII := func(r rune) bool { return r >= 0x80 }
			if bytes.IndexFunc(tr.blk[:], notASCII) >= 0 {
				hdr.Format = FormatUnknown // Non-ASCII characters in block.
			}
			nul := func(b []byte) bool { return int(b[len(b)-1]) == 0 }
			if !(nul(v7.size()) && nul(v7.mode()) && nul(v7.uid()) && nul(v7.gid()) &&
				nul(v7.modTime()) && nul(ustar.devMajor()) && nul(ustar.devMinor())) {
				hdr.Format = FormatUnknown // Numeric fields must end in NUL
			}
		case format.has(formatSTAR):
			star := tr.blk.toSTAR()
			prefix = p.parseString(star.prefix())
			hdr.AccessTime = time.Unix(p.parseNumeric(star.accessTime()), 0)
			hdr.ChangeTime = time.Unix(p.parseNumeric(star.changeTime()), 0)
		case format.has(FormatGNU):
			hdr.Format = format
			var p2 parser
			gnu := tr.blk.toGNU()
			if b := gnu.accessTime(); b[0] != 0 {
				hdr.AccessTime = time.Unix(p2.parseNumeric(b), 0)
			}
			if b := gnu.changeTime(); b[0] != 0 {
				hdr.ChangeTime = time.Unix(p2.parseNumeric(b), 0)
			}

			// Prior to Go1.8, the Writer had a bug where it would output
			// an invalid tar file in certain rare situations because the logic
			// incorrectly believed that the old GNU format had a prefix field.
			// This is wrong and leads to an output file that mangles the
			// atime and ctime fields, which are often left unused.
			//
			// In order to continue reading tar files created by former, buggy
			// versions of Go, we skeptically parse the atime and ctime fields.
			// If we are unable to parse them and the prefix field looks like
			// an ASCII string, then we fallback on the pre-Go1.8 behavior
			// of treating these fields as the USTAR prefix field.
			//
			// Note that this will not use the fallback logic for all possible
			// files generated by a pre-Go1.8 toolchain. If the generated file
			// happened to have a prefix field that parses as valid
			// atime and ctime fields (e.g., when they are valid octal strings),
			// then it is impossible to distinguish between a valid GNU file
			// and an invalid pre-Go1.8 file.
			//
			// See https://golang.org/issues/12594
			// See https://golang.org/issues/21005
			if p2.err != nil {
				hdr.AccessTime, hdr.ChangeTime = time.Time{}, time.Time{}
				ustar := tr.blk.toUSTAR()
				if s := p.parseString(ustar.prefix()); isASCII(s) {
					prefix = s
				}
				hdr.Format = FormatUnknown // Buggy file is not GNU
			}
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
func (tr *Reader) readOldGNUSparseMap(hdr *Header, blk *block) (sparseDatas, error) {
	// Make sure that the input format is GNU.
	// Unfortunately, the STAR format also has a sparse header format that uses
	// the same type flag but has a completely different layout.
	if blk.getFormat() != FormatGNU {
		return nil, ErrHeader
	}
	hdr.Format.mayOnlyBe(FormatGNU)

	var p parser
	hdr.Size = p.parseNumeric(blk.toGNU().realSize())
	if p.err != nil {
		return nil, p.err
	}
	s := blk.toGNU().sparse()
	spd := make(sparseDatas, 0, s.maxEntries())
	for {
		for i := 0; i < s.maxEntries(); i++ {
			// This termination condition is identical to GNU and BSD tar.
			if s.entry(i).offset()[0] == 0x00 {
				break // Don't return, need to process extended headers (even if empty)
			}
			offset := p.parseNumeric(s.entry(i).offset())
			length := p.parseNumeric(s.entry(i).length())
			if p.err != nil {
				return nil, p.err
			}
			spd = append(spd, sparseEntry{Offset: offset, Length: length})
		}

		if s.isExtended()[0] > 0 {
			// There are more entries. Read an extension header and parse its entries.
			if _, err := mustReadFull(tr.r, blk[:]); err != nil {
				return nil, err
			}
			s = blk.toSparse()
			continue
		}
		return spd, nil // Done
	}
}

// readGNUSparseMap1x0 reads the sparse map as stored in GNU's PAX sparse format
// version 1.0. The format of the sparse map consists of a series of
// newline-terminated numeric fields. The first field is the number of entries
// and is always present. Following this are the entries, consisting of two
// fields (offset, length). This function must stop reading at the end
// boundary of the block containing the last newline.
//
// Note that the GNU manual says that numeric values should be encoded in octal
// format. However, the GNU tar utility itself outputs these values in decimal.
// As such, this library treats values as being encoded in decimal.
func readGNUSparseMap1x0(r io.Reader) (sparseDatas, error) {
	var (
		cntNewline int64
		buf        bytes.Buffer
		blk        block
	)

	// feedTokens copies data in blocks from r into buf until there are
	// at least cnt newlines in buf. It will not read more blocks than needed.
	feedTokens := func(n int64) error {
		for cntNewline < n {
			if _, err := mustReadFull(r, blk[:]); err != nil {
				return err
			}
			buf.Write(blk[:])
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
	nextToken := func() string {
		cntNewline--
		tok, _ := buf.ReadString('\n')
		return strings.TrimRight(tok, "\n")
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
	spd := make(sparseDatas, 0, numEntries)
	for i := int64(0); i < numEntries; i++ {
		offset, err1 := strconv.ParseInt(nextToken(), 10, 64)
		length, err2 := strconv.ParseInt(nextToken(), 10, 64)
		if err1 != nil || err2 != nil {
			return nil, ErrHeader
		}
		spd = append(spd, sparseEntry{Offset: offset, Length: length})
	}
	return spd, nil
}

// readGNUSparseMap0x1 reads the sparse map as stored in GNU's PAX sparse format
// version 0.1. The sparse map is stored in the PAX headers.
func readGNUSparseMap0x1(paxHdrs map[string]string) (sparseDatas, error) {
	// Get number of entries.
	// Use integer overflow resistant math to check this.
	numEntriesStr := paxHdrs[paxGNUSparseNumBlocks]
	numEntries, err := strconv.ParseInt(numEntriesStr, 10, 0) // Intentionally parse as native int
	if err != nil || numEntries < 0 || int(2*numEntries) < int(numEntries) {
		return nil, ErrHeader
	}

	// There should be two numbers in sparseMap for each entry.
	sparseMap := strings.Split(paxHdrs[paxGNUSparseMap], ",")
	if len(sparseMap) == 1 && sparseMap[0] == "" {
		sparseMap = sparseMap[:0]
	}
	if int64(len(sparseMap)) != 2*numEntries {
		return nil, ErrHeader
	}

	// Loop through the entries in the sparse map.
	// numEntries is trusted now.
	spd := make(sparseDatas, 0, numEntries)
	for len(sparseMap) >= 2 {
		offset, err1 := strconv.ParseInt(sparseMap[0], 10, 64)
		length, err2 := strconv.ParseInt(sparseMap[1], 10, 64)
		if err1 != nil || err2 != nil {
			return nil, ErrHeader
		}
		spd = append(spd, sparseEntry{Offset: offset, Length: length})
		sparseMap = sparseMap[2:]
	}
	return spd, nil
}

// Read reads from the current file in the tar archive.
// It returns (0, io.EOF) when it reaches the end of that file,
// until Next is called to advance to the next file.
//
// If the current file is sparse, then the regions marked as a hole
// are read back as NUL-bytes.
//
// Calling Read on special types like TypeLink, TypeSymlink, TypeChar,
// TypeBlock, TypeDir, and TypeFifo returns (0, io.EOF) regardless of what
// the Header.Size claims.
func (tr *Reader) Read(b []byte) (int, error) {
	if tr.err != nil {
		return 0, tr.err
	}
	n, err := tr.curr.Read(b)
	if err != nil && err != io.EOF {
		tr.err = err
	}
	return n, err
}

// writeTo writes the content of the current file to w.
// The bytes written matches the number of remaining bytes in the current file.
//
// If the current file is sparse and w is an io.WriteSeeker,
// then writeTo uses Seek to skip past holes defined in Header.SparseHoles,
// assuming that skipped regions are filled with NULs.
// This always writes the last byte to ensure w is the right size.
//
// TODO(dsnet): Re-export this when adding sparse file support.
// See https://golang.org/issue/22735
func (tr *Reader) writeTo(w io.Writer) (int64, error) {
	if tr.err != nil {
		return 0, tr.err
	}
	n, err := tr.curr.WriteTo(w)
	if err != nil {
		tr.err = err
	}
	return n, err
}

// regFileReader is a fileReader for reading data from a regular file entry.
type regFileReader struct {
	r  io.Reader // Underlying Reader
	nb int64     // Number of remaining bytes to read
}

func (fr *regFileReader) Read(b []byte) (n int, err error) {
	if int64(len(b)) > fr.nb {
		b = b[:fr.nb]
	}
	if len(b) > 0 {
		n, err = fr.r.Read(b)
		fr.nb -= int64(n)
	}
	switch {
	case err == io.EOF && fr.nb > 0:
		return n, io.ErrUnexpectedEOF
	case err == nil && fr.nb == 0:
		return n, io.EOF
	default:
		return n, err
	}
}

func (fr *regFileReader) WriteTo(w io.Writer) (int64, error) {
	return io.Copy(w, struct{ io.Reader }{fr})
}

// logicalRemaining implements fileState.logicalRemaining.
func (fr regFileReader) logicalRemaining() int64 {
	return fr.nb
}

// logicalRemaining implements fileState.physicalRemaining.
func (fr regFileReader) physicalRemaining() int64 {
	return fr.nb
}

// sparseFileReader is a fileReader for reading data from a sparse file entry.
type sparseFileReader struct {
	fr  fileReader  // Underlying fileReader
	sp  sparseHoles // Normalized list of sparse holes
	pos int64       // Current position in sparse file
}

func (sr *sparseFileReader) Read(b []byte) (n int, err error) {
	finished := int64(len(b)) >= sr.logicalRemaining()
	if finished {
		b = b[:sr.logicalRemaining()]
	}

	b0 := b
	endPos := sr.pos + int64(len(b))
	for endPos > sr.pos && err == nil {
		var nf int // Bytes read in fragment
		holeStart, holeEnd := sr.sp[0].Offset, sr.sp[0].endOffset()
		if sr.pos < holeStart { // In a data fragment
			bf := b[:min(int64(len(b)), holeStart-sr.pos)]
			nf, err = tryReadFull(sr.fr, bf)
		} else { // In a hole fragment
			bf := b[:min(int64(len(b)), holeEnd-sr.pos)]
			nf, err = tryReadFull(zeroReader{}, bf)
		}
		b = b[nf:]
		sr.pos += int64(nf)
		if sr.pos >= holeEnd && len(sr.sp) > 1 {
			sr.sp = sr.sp[1:] // Ensure last fragment always remains
		}
	}

	n = len(b0) - len(b)
	switch {
	case err == io.EOF:
		return n, errMissData // Less data in dense file than sparse file
	case err != nil:
		return n, err
	case sr.logicalRemaining() == 0 && sr.physicalRemaining() > 0:
		return n, errUnrefData // More data in dense file than sparse file
	case finished:
		return n, io.EOF
	default:
		return n, nil
	}
}

func (sr *sparseFileReader) WriteTo(w io.Writer) (n int64, err error) {
	ws, ok := w.(io.WriteSeeker)
	if ok {
		if _, err := ws.Seek(0, io.SeekCurrent); err != nil {
			ok = false // Not all io.Seeker can really seek
		}
	}
	if !ok {
		return io.Copy(w, struct{ io.Reader }{sr})
	}

	var writeLastByte bool
	pos0 := sr.pos
	for sr.logicalRemaining() > 0 && !writeLastByte && err == nil {
		var nf int64 // Size of fragment
		holeStart, holeEnd := sr.sp[0].Offset, sr.sp[0].endOffset()
		if sr.pos < holeStart { // In a data fragment
			nf = holeStart - sr.pos
			nf, err = io.CopyN(ws, sr.fr, nf)
		} else { // In a hole fragment
			nf = holeEnd - sr.pos
			if sr.physicalRemaining() == 0 {
				writeLastByte = true
				nf--
			}
			_, err = ws.Seek(nf, io.SeekCurrent)
		}
		sr.pos += nf
		if sr.pos >= holeEnd && len(sr.sp) > 1 {
			sr.sp = sr.sp[1:] // Ensure last fragment always remains
		}
	}

	// If the last fragment is a hole, then seek to 1-byte before EOF, and
	// write a single byte to ensure the file is the right size.
	if writeLastByte && err == nil {
		_, err = ws.Write([]byte{0})
		sr.pos++
	}

	n = sr.pos - pos0
	switch {
	case err == io.EOF:
		return n, errMissData // Less data in dense file than sparse file
	case err != nil:
		return n, err
	case sr.logicalRemaining() == 0 && sr.physicalRemaining() > 0:
		return n, errUnrefData // More data in dense file than sparse file
	default:
		return n, nil
	}
}

func (sr sparseFileReader) logicalRemaining() int64 {
	return sr.sp[len(sr.sp)-1].endOffset() - sr.pos
}
func (sr sparseFileReader) physicalRemaining() int64 {
	return sr.fr.physicalRemaining()
}

type zeroReader struct{}

func (zeroReader) Read(b []byte) (int, error) {
	for i := range b {
		b[i] = 0
	}
	return len(b), nil
}

// mustReadFull is like io.ReadFull except it returns
// io.ErrUnexpectedEOF when io.EOF is hit before len(b) bytes are read.
func mustReadFull(r io.Reader, b []byte) (int, error) {
	n, err := tryReadFull(r, b)
	if err == io.EOF {
		err = io.ErrUnexpectedEOF
	}
	return n, err
}

// tryReadFull is like io.ReadFull except it returns
// io.EOF when it is hit before len(b) bytes are read.
func tryReadFull(r io.Reader, b []byte) (n int, err error) {
	for len(b) > n && err == nil {
		var nn int
		nn, err = r.Read(b[n:])
		n += nn
	}
	if len(b) == n && err == io.EOF {
		err = nil
	}
	return n, err
}

// readSpecialFile is like io.ReadAll except it returns
// ErrFieldTooLong if more than maxSpecialFileSize is read.
func readSpecialFile(r io.Reader) ([]byte, error) {
	buf, err := io.ReadAll(io.LimitReader(r, maxSpecialFileSize+1))
	if len(buf) > maxSpecialFileSize {
		return nil, ErrFieldTooLong
	}
	return buf, err
}

// discard skips n bytes in r, reporting an error if unable to do so.
func discard(r io.Reader, n int64) error {
	// If possible, Seek to the last byte before the end of the data section.
	// Do this because Seek is often lazy about reporting errors; this will mask
	// the fact that the stream may be truncated. We can rely on the
	// io.CopyN done shortly afterwards to trigger any IO errors.
	var seekSkipped int64 // Number of bytes skipped via Seek
	if sr, ok := r.(io.Seeker); ok && n > 1 {
		// Not all io.Seeker can actually Seek. For example, os.Stdin implements
		// io.Seeker, but calling Seek always returns an error and performs
		// no action. Thus, we try an innocent seek to the current position
		// to see if Seek is really supported.
		pos1, err := sr.Seek(0, io.SeekCurrent)
		if pos1 >= 0 && err == nil {
			// Seek seems supported, so perform the real Seek.
			pos2, err := sr.Seek(n-1, io.SeekCurrent)
			if pos2 < 0 || err != nil {
				return err
			}
			seekSkipped = pos2 - pos1
		}
	}

	copySkipped, err := io.CopyN(io.Discard, r, n-seekSkipped)
	if err == io.EOF && seekSkipped+copySkipped < n {
		err = io.ErrUnexpectedEOF
	}
	return err
}
