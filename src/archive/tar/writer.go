// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"path"
	"slices"
	"strings"
	"time"
)

// Writer provides sequential writing of a tar archive.
// [Writer.WriteHeader] begins a new file with the provided [Header],
// and then Writer can be treated as an io.Writer to supply that file's data.
type Writer struct {
	w    io.Writer
	pad  int64      // Amount of padding to write after current file entry
	curr fileWriter // Writer for current file entry
	hdr  Header     // Shallow copy of Header that is safe for mutations
	blk  block      // Buffer to use as temporary local storage

	// err is a persistent error.
	// It is only the responsibility of every exported method of Writer to
	// ensure that this error is sticky.
	err error
}

// NewWriter creates a new Writer writing to w.
func NewWriter(w io.Writer) *Writer {
	return &Writer{w: w, curr: &regFileWriter{w, 0}}
}

type fileWriter interface {
	io.Writer
	fileState

	ReadFrom(io.Reader) (int64, error)
}

// Flush finishes writing the current file's block padding.
// The current file must be fully written before Flush can be called.
//
// This is unnecessary as the next call to [Writer.WriteHeader] or [Writer.Close]
// will implicitly flush out the file's padding.
func (tw *Writer) Flush() error {
	if tw.err != nil {
		return tw.err
	}
	if nb := tw.curr.logicalRemaining(); nb > 0 {
		return fmt.Errorf("archive/tar: missed writing %d bytes", nb)
	}
	if _, tw.err = tw.w.Write(zeroBlock[:tw.pad]); tw.err != nil {
		return tw.err
	}
	tw.pad = 0
	return nil
}

// WriteHeader writes hdr and prepares to accept the file's contents.
// The Header.Size determines how many bytes can be written for the next file.
// If the current file is not fully written, then this returns an error.
// This implicitly flushes any padding necessary before writing the header.
func (tw *Writer) WriteHeader(hdr *Header) error {
	if err := tw.Flush(); err != nil {
		return err
	}
	tw.hdr = *hdr // Shallow copy of Header

	// Avoid usage of the legacy TypeRegA flag, and automatically promote
	// it to use TypeReg or TypeDir.
	if tw.hdr.Typeflag == TypeRegA {
		if strings.HasSuffix(tw.hdr.Name, "/") {
			tw.hdr.Typeflag = TypeDir
		} else {
			tw.hdr.Typeflag = TypeReg
		}
	}

	// Round ModTime and ignore AccessTime and ChangeTime unless
	// the format is explicitly chosen.
	// This ensures nominal usage of WriteHeader (without specifying the format)
	// does not always result in the PAX format being chosen, which
	// causes a 1KiB increase to every header.
	if tw.hdr.Format == FormatUnknown {
		tw.hdr.ModTime = tw.hdr.ModTime.Round(time.Second)
		tw.hdr.AccessTime = time.Time{}
		tw.hdr.ChangeTime = time.Time{}
	}

	allowedFormats, paxHdrs, err := tw.hdr.allowedFormats()
	switch {
	case allowedFormats.has(FormatUSTAR):
		tw.err = tw.writeUSTARHeader(&tw.hdr)
		return tw.err
	case allowedFormats.has(FormatPAX):
		tw.err = tw.writePAXHeader(&tw.hdr, paxHdrs)
		return tw.err
	case allowedFormats.has(FormatGNU):
		tw.err = tw.writeGNUHeader(&tw.hdr)
		return tw.err
	default:
		return err // Non-fatal error
	}
}

func (tw *Writer) writeUSTARHeader(hdr *Header) error {
	// Check if we can use USTAR prefix/suffix splitting.
	var namePrefix string
	if prefix, suffix, ok := splitUSTARPath(hdr.Name); ok {
		namePrefix, hdr.Name = prefix, suffix
	}

	// Pack the main header.
	var f formatter
	blk := tw.templateV7Plus(hdr, f.formatString, f.formatOctal)
	f.formatString(blk.toUSTAR().prefix(), namePrefix)
	blk.setFormat(FormatUSTAR)
	if f.err != nil {
		return f.err // Should never happen since header is validated
	}
	return tw.writeRawHeader(blk, hdr.Size, hdr.Typeflag)
}

func (tw *Writer) writePAXHeader(hdr *Header, paxHdrs map[string]string) error {
	realName, realSize := hdr.Name, hdr.Size

	// TODO(dsnet): Re-enable this when adding sparse support.
	// See https://golang.org/issue/22735
	/*
		// Handle sparse files.
		var spd sparseDatas
		var spb []byte
		if len(hdr.SparseHoles) > 0 {
			sph := append([]sparseEntry{}, hdr.SparseHoles...) // Copy sparse map
			sph = alignSparseEntries(sph, hdr.Size)
			spd = invertSparseEntries(sph, hdr.Size)

			// Format the sparse map.
			hdr.Size = 0 // Replace with encoded size
			spb = append(strconv.AppendInt(spb, int64(len(spd)), 10), '\n')
			for _, s := range spd {
				hdr.Size += s.Length
				spb = append(strconv.AppendInt(spb, s.Offset, 10), '\n')
				spb = append(strconv.AppendInt(spb, s.Length, 10), '\n')
			}
			pad := blockPadding(int64(len(spb)))
			spb = append(spb, zeroBlock[:pad]...)
			hdr.Size += int64(len(spb)) // Accounts for encoded sparse map

			// Add and modify appropriate PAX records.
			dir, file := path.Split(realName)
			hdr.Name = path.Join(dir, "GNUSparseFile.0", file)
			paxHdrs[paxGNUSparseMajor] = "1"
			paxHdrs[paxGNUSparseMinor] = "0"
			paxHdrs[paxGNUSparseName] = realName
			paxHdrs[paxGNUSparseRealSize] = strconv.FormatInt(realSize, 10)
			paxHdrs[paxSize] = strconv.FormatInt(hdr.Size, 10)
			delete(paxHdrs, paxPath) // Recorded by paxGNUSparseName
		}
	*/
	_ = realSize

	// Write PAX records to the output.
	isGlobal := hdr.Typeflag == TypeXGlobalHeader
	if len(paxHdrs) > 0 || isGlobal {
		// Sort keys for deterministic ordering.
		var keys []string
		for k := range paxHdrs {
			keys = append(keys, k)
		}
		slices.Sort(keys)

		// Write each record to a buffer.
		var buf strings.Builder
		for _, k := range keys {
			rec, err := formatPAXRecord(k, paxHdrs[k])
			if err != nil {
				return err
			}
			buf.WriteString(rec)
		}

		// Write the extended header file.
		var name string
		var flag byte
		if isGlobal {
			name = realName
			if name == "" {
				name = "GlobalHead.0.0"
			}
			flag = TypeXGlobalHeader
		} else {
			dir, file := path.Split(realName)
			name = path.Join(dir, "PaxHeaders.0", file)
			flag = TypeXHeader
		}
		data := buf.String()
		if len(data) > maxSpecialFileSize {
			return ErrFieldTooLong
		}
		if err := tw.writeRawFile(name, data, flag, FormatPAX); err != nil || isGlobal {
			return err // Global headers return here
		}
	}

	// Pack the main header.
	var f formatter // Ignore errors since they are expected
	fmtStr := func(b []byte, s string) { f.formatString(b, toASCII(s)) }
	blk := tw.templateV7Plus(hdr, fmtStr, f.formatOctal)
	blk.setFormat(FormatPAX)
	if err := tw.writeRawHeader(blk, hdr.Size, hdr.Typeflag); err != nil {
		return err
	}

	// TODO(dsnet): Re-enable this when adding sparse support.
	// See https://golang.org/issue/22735
	/*
		// Write the sparse map and setup the sparse writer if necessary.
		if len(spd) > 0 {
			// Use tw.curr since the sparse map is accounted for in hdr.Size.
			if _, err := tw.curr.Write(spb); err != nil {
				return err
			}
			tw.curr = &sparseFileWriter{tw.curr, spd, 0}
		}
	*/
	return nil
}

func (tw *Writer) writeGNUHeader(hdr *Header) error {
	// Use long-link files if Name or Linkname exceeds the field size.
	const longName = "././@LongLink"
	if len(hdr.Name) > nameSize {
		data := hdr.Name + "\x00"
		if err := tw.writeRawFile(longName, data, TypeGNULongName, FormatGNU); err != nil {
			return err
		}
	}
	if len(hdr.Linkname) > nameSize {
		data := hdr.Linkname + "\x00"
		if err := tw.writeRawFile(longName, data, TypeGNULongLink, FormatGNU); err != nil {
			return err
		}
	}

	// Pack the main header.
	var f formatter // Ignore errors since they are expected
	var spd sparseDatas
	var spb []byte
	blk := tw.templateV7Plus(hdr, f.formatString, f.formatNumeric)
	if !hdr.AccessTime.IsZero() {
		f.formatNumeric(blk.toGNU().accessTime(), hdr.AccessTime.Unix())
	}
	if !hdr.ChangeTime.IsZero() {
		f.formatNumeric(blk.toGNU().changeTime(), hdr.ChangeTime.Unix())
	}
	// TODO(dsnet): Re-enable this when adding sparse support.
	// See https://golang.org/issue/22735
	/*
		if hdr.Typeflag == TypeGNUSparse {
			sph := append([]sparseEntry{}, hdr.SparseHoles...) // Copy sparse map
			sph = alignSparseEntries(sph, hdr.Size)
			spd = invertSparseEntries(sph, hdr.Size)

			// Format the sparse map.
			formatSPD := func(sp sparseDatas, sa sparseArray) sparseDatas {
				for i := 0; len(sp) > 0 && i < sa.MaxEntries(); i++ {
					f.formatNumeric(sa.Entry(i).Offset(), sp[0].Offset)
					f.formatNumeric(sa.Entry(i).Length(), sp[0].Length)
					sp = sp[1:]
				}
				if len(sp) > 0 {
					sa.IsExtended()[0] = 1
				}
				return sp
			}
			sp2 := formatSPD(spd, blk.GNU().Sparse())
			for len(sp2) > 0 {
				var spHdr block
				sp2 = formatSPD(sp2, spHdr.Sparse())
				spb = append(spb, spHdr[:]...)
			}

			// Update size fields in the header block.
			realSize := hdr.Size
			hdr.Size = 0 // Encoded size; does not account for encoded sparse map
			for _, s := range spd {
				hdr.Size += s.Length
			}
			copy(blk.V7().Size(), zeroBlock[:]) // Reset field
			f.formatNumeric(blk.V7().Size(), hdr.Size)
			f.formatNumeric(blk.GNU().RealSize(), realSize)
		}
	*/
	blk.setFormat(FormatGNU)
	if err := tw.writeRawHeader(blk, hdr.Size, hdr.Typeflag); err != nil {
		return err
	}

	// Write the extended sparse map and setup the sparse writer if necessary.
	if len(spd) > 0 {
		// Use tw.w since the sparse map is not accounted for in hdr.Size.
		if _, err := tw.w.Write(spb); err != nil {
			return err
		}
		tw.curr = &sparseFileWriter{tw.curr, spd, 0}
	}
	return nil
}

type (
	stringFormatter func([]byte, string)
	numberFormatter func([]byte, int64)
)

// templateV7Plus fills out the V7 fields of a block using values from hdr.
// It also fills out fields (uname, gname, devmajor, devminor) that are
// shared in the USTAR, PAX, and GNU formats using the provided formatters.
//
// The block returned is only valid until the next call to
// templateV7Plus or writeRawFile.
func (tw *Writer) templateV7Plus(hdr *Header, fmtStr stringFormatter, fmtNum numberFormatter) *block {
	tw.blk.reset()

	modTime := hdr.ModTime
	if modTime.IsZero() {
		modTime = time.Unix(0, 0)
	}

	v7 := tw.blk.toV7()
	v7.typeFlag()[0] = hdr.Typeflag
	fmtStr(v7.name(), hdr.Name)
	fmtStr(v7.linkName(), hdr.Linkname)
	fmtNum(v7.mode(), hdr.Mode)
	fmtNum(v7.uid(), int64(hdr.Uid))
	fmtNum(v7.gid(), int64(hdr.Gid))
	fmtNum(v7.size(), hdr.Size)
	fmtNum(v7.modTime(), modTime.Unix())

	ustar := tw.blk.toUSTAR()
	fmtStr(ustar.userName(), hdr.Uname)
	fmtStr(ustar.groupName(), hdr.Gname)
	fmtNum(ustar.devMajor(), hdr.Devmajor)
	fmtNum(ustar.devMinor(), hdr.Devminor)

	return &tw.blk
}

// writeRawFile writes a minimal file with the given name and flag type.
// It uses format to encode the header format and will write data as the body.
// It uses default values for all of the other fields (as BSD and GNU tar does).
func (tw *Writer) writeRawFile(name, data string, flag byte, format Format) error {
	tw.blk.reset()

	// Best effort for the filename.
	name = toASCII(name)
	if len(name) > nameSize {
		name = name[:nameSize]
	}
	name = strings.TrimRight(name, "/")

	var f formatter
	v7 := tw.blk.toV7()
	v7.typeFlag()[0] = flag
	f.formatString(v7.name(), name)
	f.formatOctal(v7.mode(), 0)
	f.formatOctal(v7.uid(), 0)
	f.formatOctal(v7.gid(), 0)
	f.formatOctal(v7.size(), int64(len(data))) // Must be < 8GiB
	f.formatOctal(v7.modTime(), 0)
	tw.blk.setFormat(format)
	if f.err != nil {
		return f.err // Only occurs if size condition is violated
	}

	// Write the header and data.
	if err := tw.writeRawHeader(&tw.blk, int64(len(data)), flag); err != nil {
		return err
	}
	_, err := io.WriteString(tw, data)
	return err
}

// writeRawHeader writes the value of blk, regardless of its value.
// It sets up the Writer such that it can accept a file of the given size.
// If the flag is a special header-only flag, then the size is treated as zero.
func (tw *Writer) writeRawHeader(blk *block, size int64, flag byte) error {
	if err := tw.Flush(); err != nil {
		return err
	}
	if _, err := tw.w.Write(blk[:]); err != nil {
		return err
	}
	if isHeaderOnlyType(flag) {
		size = 0
	}
	tw.curr = &regFileWriter{tw.w, size}
	tw.pad = blockPadding(size)
	return nil
}

// AddFS adds the files from fs.FS to the archive.
// It walks the directory tree starting at the root of the filesystem
// adding each file to the tar archive while maintaining the directory structure.
func (tw *Writer) AddFS(fsys fs.FS) error {
	return fs.WalkDir(fsys, ".", func(name string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		// TODO(#49580): Handle symlinks when fs.ReadLinkFS is available.
		if !info.Mode().IsRegular() {
			return errors.New("tar: cannot add non-regular file")
		}
		h, err := FileInfoHeader(info, "")
		if err != nil {
			return err
		}
		h.Name = name
		if err := tw.WriteHeader(h); err != nil {
			return err
		}
		f, err := fsys.Open(name)
		if err != nil {
			return err
		}
		defer f.Close()
		_, err = io.Copy(tw, f)
		return err
	})
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

// Write writes to the current file in the tar archive.
// Write returns the error [ErrWriteTooLong] if more than
// Header.Size bytes are written after [Writer.WriteHeader].
//
// Calling Write on special types like [TypeLink], [TypeSymlink], [TypeChar],
// [TypeBlock], [TypeDir], and [TypeFifo] returns (0, [ErrWriteTooLong]) regardless
// of what the [Header.Size] claims.
func (tw *Writer) Write(b []byte) (int, error) {
	if tw.err != nil {
		return 0, tw.err
	}
	n, err := tw.curr.Write(b)
	if err != nil && !errors.Is(err, ErrWriteTooLong) {
		tw.err = err
	}
	return n, err
}

// readFrom populates the content of the current file by reading from r.
// The bytes read must match the number of remaining bytes in the current file.
//
// If the current file is sparse and r is an io.ReadSeeker,
// then readFrom uses Seek to skip past holes defined in Header.SparseHoles,
// assuming that skipped regions are all NULs.
// This always reads the last byte to ensure r is the right size.
//
// TODO(dsnet): Re-export this when adding sparse file support.
// See https://golang.org/issue/22735
func (tw *Writer) readFrom(r io.Reader) (int64, error) {
	if tw.err != nil {
		return 0, tw.err
	}
	n, err := tw.curr.ReadFrom(r)
	if err != nil && !errors.Is(err, ErrWriteTooLong) {
		tw.err = err
	}
	return n, err
}

// Close closes the tar archive by flushing the padding, and writing the footer.
// If the current file (from a prior call to [Writer.WriteHeader]) is not fully written,
// then this returns an error.
func (tw *Writer) Close() error {
	if errors.Is(tw.err, ErrWriteAfterClose) {
		return nil
	}
	if tw.err != nil {
		return tw.err
	}

	// Trailer: two zero blocks.
	err := tw.Flush()
	for i := 0; i < 2 && err == nil; i++ {
		_, err = tw.w.Write(zeroBlock[:])
	}

	// Ensure all future actions are invalid.
	tw.err = ErrWriteAfterClose
	return err // Report IO errors
}

// regFileWriter is a fileWriter for writing data to a regular file entry.
type regFileWriter struct {
	w  io.Writer // Underlying Writer
	nb int64     // Number of remaining bytes to write
}

func (fw *regFileWriter) Write(b []byte) (n int, err error) {
	overwrite := int64(len(b)) > fw.nb
	if overwrite {
		b = b[:fw.nb]
	}
	if len(b) > 0 {
		n, err = fw.w.Write(b)
		fw.nb -= int64(n)
	}
	switch {
	case err != nil:
		return n, err
	case overwrite:
		return n, ErrWriteTooLong
	default:
		return n, nil
	}
}

func (fw *regFileWriter) ReadFrom(r io.Reader) (int64, error) {
	return io.Copy(struct{ io.Writer }{fw}, r)
}

// logicalRemaining implements fileState.logicalRemaining.
func (fw regFileWriter) logicalRemaining() int64 {
	return fw.nb
}

// physicalRemaining implements fileState.physicalRemaining.
func (fw regFileWriter) physicalRemaining() int64 {
	return fw.nb
}

// sparseFileWriter is a fileWriter for writing data to a sparse file entry.
type sparseFileWriter struct {
	fw  fileWriter  // Underlying fileWriter
	sp  sparseDatas // Normalized list of data fragments
	pos int64       // Current position in sparse file
}

func (sw *sparseFileWriter) Write(b []byte) (n int, err error) {
	overwrite := int64(len(b)) > sw.logicalRemaining()
	if overwrite {
		b = b[:sw.logicalRemaining()]
	}

	b0 := b
	endPos := sw.pos + int64(len(b))
	for endPos > sw.pos && err == nil {
		var nf int // Bytes written in fragment
		dataStart, dataEnd := sw.sp[0].Offset, sw.sp[0].endOffset()
		if sw.pos < dataStart { // In a hole fragment
			bf := b[:min(int64(len(b)), dataStart-sw.pos)]
			nf, err = zeroWriter{}.Write(bf)
		} else { // In a data fragment
			bf := b[:min(int64(len(b)), dataEnd-sw.pos)]
			nf, err = sw.fw.Write(bf)
		}
		b = b[nf:]
		sw.pos += int64(nf)
		if sw.pos >= dataEnd && len(sw.sp) > 1 {
			sw.sp = sw.sp[1:] // Ensure last fragment always remains
		}
	}

	n = len(b0) - len(b)
	switch {
	case errors.Is(err, ErrWriteTooLong):
		return n, errMissData // Not possible; implies bug in validation logic
	case err != nil:
		return n, err
	case sw.logicalRemaining() == 0 && sw.physicalRemaining() > 0:
		return n, errUnrefData // Not possible; implies bug in validation logic
	case overwrite:
		return n, ErrWriteTooLong
	default:
		return n, nil
	}
}

func (sw *sparseFileWriter) ReadFrom(r io.Reader) (n int64, err error) {
	rs, ok := r.(io.ReadSeeker)
	if ok {
		if _, err := rs.Seek(0, io.SeekCurrent); err != nil {
			ok = false // Not all io.Seeker can really seek
		}
	}
	if !ok {
		return io.Copy(struct{ io.Writer }{sw}, r)
	}

	var readLastByte bool
	pos0 := sw.pos
	for sw.logicalRemaining() > 0 && !readLastByte && err == nil {
		var nf int64 // Size of fragment
		dataStart, dataEnd := sw.sp[0].Offset, sw.sp[0].endOffset()
		if sw.pos < dataStart { // In a hole fragment
			nf = dataStart - sw.pos
			if sw.physicalRemaining() == 0 {
				readLastByte = true
				nf--
			}
			_, err = rs.Seek(nf, io.SeekCurrent)
		} else { // In a data fragment
			nf = dataEnd - sw.pos
			nf, err = io.CopyN(sw.fw, rs, nf)
		}
		sw.pos += nf
		if sw.pos >= dataEnd && len(sw.sp) > 1 {
			sw.sp = sw.sp[1:] // Ensure last fragment always remains
		}
	}

	// If the last fragment is a hole, then seek to 1-byte before EOF, and
	// read a single byte to ensure the file is the right size.
	if readLastByte && err == nil {
		_, err = mustReadFull(rs, []byte{0})
		sw.pos++
	}

	n = sw.pos - pos0
	switch {
	case err == io.EOF:
		return n, io.ErrUnexpectedEOF
	case errors.Is(err, ErrWriteTooLong):
		return n, errMissData // Not possible; implies bug in validation logic
	case err != nil:
		return n, err
	case sw.logicalRemaining() == 0 && sw.physicalRemaining() > 0:
		return n, errUnrefData // Not possible; implies bug in validation logic
	default:
		return n, ensureEOF(rs)
	}
}

func (sw sparseFileWriter) logicalRemaining() int64 {
	return sw.sp[len(sw.sp)-1].endOffset() - sw.pos
}
func (sw sparseFileWriter) physicalRemaining() int64 {
	return sw.fw.physicalRemaining()
}

// zeroWriter may only be written with NULs, otherwise it returns errWriteHole.
type zeroWriter struct{}

func (zeroWriter) Write(b []byte) (int, error) {
	for i, c := range b {
		if c != 0 {
			return i, errWriteHole
		}
	}
	return len(b), nil
}

// ensureEOF checks whether r is at EOF, reporting ErrWriteTooLong if not so.
func ensureEOF(r io.Reader) error {
	n, err := tryReadFull(r, []byte{0})
	switch {
	case n > 0:
		return ErrWriteTooLong
	case err == io.EOF:
		return nil
	default:
		return err
	}
}
