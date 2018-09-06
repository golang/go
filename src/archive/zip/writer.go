// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zip

import (
	"bufio"
	"encoding/binary"
	"errors"
	"hash"
	"hash/crc32"
	"io"
	"strings"
	"unicode/utf8"
)

var (
	errLongName  = errors.New("zip: FileHeader.Name too long")
	errLongExtra = errors.New("zip: FileHeader.Extra too long")
)

// Writer implements a zip file writer.
type Writer struct {
	cw          *countWriter
	dir         []*header
	last        *fileWriter
	closed      bool
	compressors map[uint16]Compressor
	comment     string

	// testHookCloseSizeOffset if non-nil is called with the size
	// of offset of the central directory at Close.
	testHookCloseSizeOffset func(size, offset uint64)
}

type header struct {
	*FileHeader
	offset uint64
}

// NewWriter returns a new Writer writing a zip file to w.
func NewWriter(w io.Writer) *Writer {
	return &Writer{cw: &countWriter{w: bufio.NewWriter(w)}}
}

// SetOffset sets the offset of the beginning of the zip data within the
// underlying writer. It should be used when the zip data is appended to an
// existing file, such as a binary executable.
// It must be called before any data is written.
func (w *Writer) SetOffset(n int64) {
	if w.cw.count != 0 {
		panic("zip: SetOffset called after data was written")
	}
	w.cw.count = n
}

// Flush flushes any buffered data to the underlying writer.
// Calling Flush is not normally necessary; calling Close is sufficient.
func (w *Writer) Flush() error {
	return w.cw.w.(*bufio.Writer).Flush()
}

// SetComment sets the end-of-central-directory comment field.
// It can only be called before Close.
func (w *Writer) SetComment(comment string) error {
	if len(comment) > uint16max {
		return errors.New("zip: Writer.Comment too long")
	}
	w.comment = comment
	return nil
}

// Close finishes writing the zip file by writing the central directory.
// It does not close the underlying writer.
func (w *Writer) Close() error {
	if w.last != nil && !w.last.closed {
		if err := w.last.close(); err != nil {
			return err
		}
		w.last = nil
	}
	if w.closed {
		return errors.New("zip: writer closed twice")
	}
	w.closed = true

	// write central directory
	start := w.cw.count
	for _, h := range w.dir {
		var buf [directoryHeaderLen]byte
		b := writeBuf(buf[:])
		b.uint32(uint32(directoryHeaderSignature))
		b.uint16(h.CreatorVersion)
		b.uint16(h.ReaderVersion)
		b.uint16(h.Flags)
		b.uint16(h.Method)
		b.uint16(h.ModifiedTime)
		b.uint16(h.ModifiedDate)
		b.uint32(h.CRC32)
		if h.isZip64() || h.offset >= uint32max {
			// the file needs a zip64 header. store maxint in both
			// 32 bit size fields (and offset later) to signal that the
			// zip64 extra header should be used.
			b.uint32(uint32max) // compressed size
			b.uint32(uint32max) // uncompressed size

			// append a zip64 extra block to Extra
			var buf [28]byte // 2x uint16 + 3x uint64
			eb := writeBuf(buf[:])
			eb.uint16(zip64ExtraID)
			eb.uint16(24) // size = 3x uint64
			eb.uint64(h.UncompressedSize64)
			eb.uint64(h.CompressedSize64)
			eb.uint64(h.offset)
			h.Extra = append(h.Extra, buf[:]...)
		} else {
			b.uint32(h.CompressedSize)
			b.uint32(h.UncompressedSize)
		}

		b.uint16(uint16(len(h.Name)))
		b.uint16(uint16(len(h.Extra)))
		b.uint16(uint16(len(h.Comment)))
		b = b[4:] // skip disk number start and internal file attr (2x uint16)
		b.uint32(h.ExternalAttrs)
		if h.offset > uint32max {
			b.uint32(uint32max)
		} else {
			b.uint32(uint32(h.offset))
		}
		if _, err := w.cw.Write(buf[:]); err != nil {
			return err
		}
		if _, err := io.WriteString(w.cw, h.Name); err != nil {
			return err
		}
		if _, err := w.cw.Write(h.Extra); err != nil {
			return err
		}
		if _, err := io.WriteString(w.cw, h.Comment); err != nil {
			return err
		}
	}
	end := w.cw.count

	records := uint64(len(w.dir))
	size := uint64(end - start)
	offset := uint64(start)

	if f := w.testHookCloseSizeOffset; f != nil {
		f(size, offset)
	}

	if records >= uint16max || size >= uint32max || offset >= uint32max {
		var buf [directory64EndLen + directory64LocLen]byte
		b := writeBuf(buf[:])

		// zip64 end of central directory record
		b.uint32(directory64EndSignature)
		b.uint64(directory64EndLen - 12) // length minus signature (uint32) and length fields (uint64)
		b.uint16(zipVersion45)           // version made by
		b.uint16(zipVersion45)           // version needed to extract
		b.uint32(0)                      // number of this disk
		b.uint32(0)                      // number of the disk with the start of the central directory
		b.uint64(records)                // total number of entries in the central directory on this disk
		b.uint64(records)                // total number of entries in the central directory
		b.uint64(size)                   // size of the central directory
		b.uint64(offset)                 // offset of start of central directory with respect to the starting disk number

		// zip64 end of central directory locator
		b.uint32(directory64LocSignature)
		b.uint32(0)           // number of the disk with the start of the zip64 end of central directory
		b.uint64(uint64(end)) // relative offset of the zip64 end of central directory record
		b.uint32(1)           // total number of disks

		if _, err := w.cw.Write(buf[:]); err != nil {
			return err
		}

		// store max values in the regular end record to signal that
		// that the zip64 values should be used instead
		records = uint16max
		size = uint32max
		offset = uint32max
	}

	// write end record
	var buf [directoryEndLen]byte
	b := writeBuf(buf[:])
	b.uint32(uint32(directoryEndSignature))
	b = b[4:]                        // skip over disk number and first disk number (2x uint16)
	b.uint16(uint16(records))        // number of entries this disk
	b.uint16(uint16(records))        // number of entries total
	b.uint32(uint32(size))           // size of directory
	b.uint32(uint32(offset))         // start of directory
	b.uint16(uint16(len(w.comment))) // byte size of EOCD comment
	if _, err := w.cw.Write(buf[:]); err != nil {
		return err
	}
	if _, err := io.WriteString(w.cw, w.comment); err != nil {
		return err
	}

	return w.cw.w.(*bufio.Writer).Flush()
}

// Create adds a file to the zip file using the provided name.
// It returns a Writer to which the file contents should be written.
// The file contents will be compressed using the Deflate method.
// The name must be a relative path: it must not start with a drive
// letter (e.g. C:) or leading slash, and only forward slashes are
// allowed. To create a directory instead of a file, add a trailing
// slash to the name.
// The file's contents must be written to the io.Writer before the next
// call to Create, CreateHeader, or Close.
func (w *Writer) Create(name string) (io.Writer, error) {
	header := &FileHeader{
		Name:   name,
		Method: Deflate,
	}
	return w.CreateHeader(header)
}

// detectUTF8 reports whether s is a valid UTF-8 string, and whether the string
// must be considered UTF-8 encoding (i.e., not compatible with CP-437, ASCII,
// or any other common encoding).
func detectUTF8(s string) (valid, require bool) {
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		i += size
		// Officially, ZIP uses CP-437, but many readers use the system's
		// local character encoding. Most encoding are compatible with a large
		// subset of CP-437, which itself is ASCII-like.
		//
		// Forbid 0x7e and 0x5c since EUC-KR and Shift-JIS replace those
		// characters with localized currency and overline characters.
		if r < 0x20 || r > 0x7d || r == 0x5c {
			if !utf8.ValidRune(r) || (r == utf8.RuneError && size == 1) {
				return false, false
			}
			require = true
		}
	}
	return true, require
}

// CreateHeader adds a file to the zip archive using the provided FileHeader
// for the file metadata. Writer takes ownership of fh and may mutate
// its fields. The caller must not modify fh after calling CreateHeader.
//
// This returns a Writer to which the file contents should be written.
// The file's contents must be written to the io.Writer before the next
// call to Create, CreateHeader, or Close.
func (w *Writer) CreateHeader(fh *FileHeader) (io.Writer, error) {
	if w.last != nil && !w.last.closed {
		if err := w.last.close(); err != nil {
			return nil, err
		}
	}
	if len(w.dir) > 0 && w.dir[len(w.dir)-1].FileHeader == fh {
		// See https://golang.org/issue/11144 confusion.
		return nil, errors.New("archive/zip: invalid duplicate FileHeader")
	}

	// The ZIP format has a sad state of affairs regarding character encoding.
	// Officially, the name and comment fields are supposed to be encoded
	// in CP-437 (which is mostly compatible with ASCII), unless the UTF-8
	// flag bit is set. However, there are several problems:
	//
	//	* Many ZIP readers still do not support UTF-8.
	//	* If the UTF-8 flag is cleared, several readers simply interpret the
	//	name and comment fields as whatever the local system encoding is.
	//
	// In order to avoid breaking readers without UTF-8 support,
	// we avoid setting the UTF-8 flag if the strings are CP-437 compatible.
	// However, if the strings require multibyte UTF-8 encoding and is a
	// valid UTF-8 string, then we set the UTF-8 bit.
	//
	// For the case, where the user explicitly wants to specify the encoding
	// as UTF-8, they will need to set the flag bit themselves.
	utf8Valid1, utf8Require1 := detectUTF8(fh.Name)
	utf8Valid2, utf8Require2 := detectUTF8(fh.Comment)
	switch {
	case fh.NonUTF8:
		fh.Flags &^= 0x800
	case (utf8Require1 || utf8Require2) && (utf8Valid1 && utf8Valid2):
		fh.Flags |= 0x800
	}

	fh.CreatorVersion = fh.CreatorVersion&0xff00 | zipVersion20 // preserve compatibility byte
	fh.ReaderVersion = zipVersion20

	// If Modified is set, this takes precedence over MS-DOS timestamp fields.
	if !fh.Modified.IsZero() {
		// Contrary to the FileHeader.SetModTime method, we intentionally
		// do not convert to UTC, because we assume the user intends to encode
		// the date using the specified timezone. A user may want this control
		// because many legacy ZIP readers interpret the timestamp according
		// to the local timezone.
		//
		// The timezone is only non-UTC if a user directly sets the Modified
		// field directly themselves. All other approaches sets UTC.
		fh.ModifiedDate, fh.ModifiedTime = timeToMsDosTime(fh.Modified)

		// Use "extended timestamp" format since this is what Info-ZIP uses.
		// Nearly every major ZIP implementation uses a different format,
		// but at least most seem to be able to understand the other formats.
		//
		// This format happens to be identical for both local and central header
		// if modification time is the only timestamp being encoded.
		var mbuf [9]byte // 2*SizeOf(uint16) + SizeOf(uint8) + SizeOf(uint32)
		mt := uint32(fh.Modified.Unix())
		eb := writeBuf(mbuf[:])
		eb.uint16(extTimeExtraID)
		eb.uint16(5)  // Size: SizeOf(uint8) + SizeOf(uint32)
		eb.uint8(1)   // Flags: ModTime
		eb.uint32(mt) // ModTime
		fh.Extra = append(fh.Extra, mbuf[:]...)
	}

	var (
		ow io.Writer
		fw *fileWriter
	)
	h := &header{
		FileHeader: fh,
		offset:     uint64(w.cw.count),
	}

	if strings.HasSuffix(fh.Name, "/") {
		// Set the compression method to Store to ensure data length is truly zero,
		// which the writeHeader method always encodes for the size fields.
		// This is necessary as most compression formats have non-zero lengths
		// even when compressing an empty string.
		fh.Method = Store
		fh.Flags &^= 0x8 // we will not write a data descriptor

		// Explicitly clear sizes as they have no meaning for directories.
		fh.CompressedSize = 0
		fh.CompressedSize64 = 0
		fh.UncompressedSize = 0
		fh.UncompressedSize64 = 0

		ow = dirWriter{}
	} else {
		fh.Flags |= 0x8 // we will write a data descriptor

		fw = &fileWriter{
			zipw:      w.cw,
			compCount: &countWriter{w: w.cw},
			crc32:     crc32.NewIEEE(),
		}
		comp := w.compressor(fh.Method)
		if comp == nil {
			return nil, ErrAlgorithm
		}
		var err error
		fw.comp, err = comp(fw.compCount)
		if err != nil {
			return nil, err
		}
		fw.rawCount = &countWriter{w: fw.comp}
		fw.header = h
		ow = fw
	}
	w.dir = append(w.dir, h)
	if err := writeHeader(w.cw, fh); err != nil {
		return nil, err
	}
	// If we're creating a directory, fw is nil.
	w.last = fw
	return ow, nil
}

func writeHeader(w io.Writer, h *FileHeader) error {
	const maxUint16 = 1<<16 - 1
	if len(h.Name) > maxUint16 {
		return errLongName
	}
	if len(h.Extra) > maxUint16 {
		return errLongExtra
	}

	var buf [fileHeaderLen]byte
	b := writeBuf(buf[:])
	b.uint32(uint32(fileHeaderSignature))
	b.uint16(h.ReaderVersion)
	b.uint16(h.Flags)
	b.uint16(h.Method)
	b.uint16(h.ModifiedTime)
	b.uint16(h.ModifiedDate)
	b.uint32(0) // since we are writing a data descriptor crc32,
	b.uint32(0) // compressed size,
	b.uint32(0) // and uncompressed size should be zero
	b.uint16(uint16(len(h.Name)))
	b.uint16(uint16(len(h.Extra)))
	if _, err := w.Write(buf[:]); err != nil {
		return err
	}
	if _, err := io.WriteString(w, h.Name); err != nil {
		return err
	}
	_, err := w.Write(h.Extra)
	return err
}

// RegisterCompressor registers or overrides a custom compressor for a specific
// method ID. If a compressor for a given method is not found, Writer will
// default to looking up the compressor at the package level.
func (w *Writer) RegisterCompressor(method uint16, comp Compressor) {
	if w.compressors == nil {
		w.compressors = make(map[uint16]Compressor)
	}
	w.compressors[method] = comp
}

func (w *Writer) compressor(method uint16) Compressor {
	comp := w.compressors[method]
	if comp == nil {
		comp = compressor(method)
	}
	return comp
}

type dirWriter struct{}

func (dirWriter) Write(b []byte) (int, error) {
	if len(b) == 0 {
		return 0, nil
	}
	return 0, errors.New("zip: write to directory")
}

type fileWriter struct {
	*header
	zipw      io.Writer
	rawCount  *countWriter
	comp      io.WriteCloser
	compCount *countWriter
	crc32     hash.Hash32
	closed    bool
}

func (w *fileWriter) Write(p []byte) (int, error) {
	if w.closed {
		return 0, errors.New("zip: write to closed file")
	}
	w.crc32.Write(p)
	return w.rawCount.Write(p)
}

func (w *fileWriter) close() error {
	if w.closed {
		return errors.New("zip: file closed twice")
	}
	w.closed = true
	if err := w.comp.Close(); err != nil {
		return err
	}

	// update FileHeader
	fh := w.header.FileHeader
	fh.CRC32 = w.crc32.Sum32()
	fh.CompressedSize64 = uint64(w.compCount.count)
	fh.UncompressedSize64 = uint64(w.rawCount.count)

	if fh.isZip64() {
		fh.CompressedSize = uint32max
		fh.UncompressedSize = uint32max
		fh.ReaderVersion = zipVersion45 // requires 4.5 - File uses ZIP64 format extensions
	} else {
		fh.CompressedSize = uint32(fh.CompressedSize64)
		fh.UncompressedSize = uint32(fh.UncompressedSize64)
	}

	// Write data descriptor. This is more complicated than one would
	// think, see e.g. comments in zipfile.c:putextended() and
	// http://bugs.sun.com/bugdatabase/view_bug.do?bug_id=7073588.
	// The approach here is to write 8 byte sizes if needed without
	// adding a zip64 extra in the local header (too late anyway).
	var buf []byte
	if fh.isZip64() {
		buf = make([]byte, dataDescriptor64Len)
	} else {
		buf = make([]byte, dataDescriptorLen)
	}
	b := writeBuf(buf)
	b.uint32(dataDescriptorSignature) // de-facto standard, required by OS X
	b.uint32(fh.CRC32)
	if fh.isZip64() {
		b.uint64(fh.CompressedSize64)
		b.uint64(fh.UncompressedSize64)
	} else {
		b.uint32(fh.CompressedSize)
		b.uint32(fh.UncompressedSize)
	}
	_, err := w.zipw.Write(buf)
	return err
}

type countWriter struct {
	w     io.Writer
	count int64
}

func (w *countWriter) Write(p []byte) (int, error) {
	n, err := w.w.Write(p)
	w.count += int64(n)
	return n, err
}

type nopCloser struct {
	io.Writer
}

func (w nopCloser) Close() error {
	return nil
}

type writeBuf []byte

func (b *writeBuf) uint8(v uint8) {
	(*b)[0] = v
	*b = (*b)[1:]
}

func (b *writeBuf) uint16(v uint16) {
	binary.LittleEndian.PutUint16(*b, v)
	*b = (*b)[2:]
}

func (b *writeBuf) uint32(v uint32) {
	binary.LittleEndian.PutUint32(*b, v)
	*b = (*b)[4:]
}

func (b *writeBuf) uint64(v uint64) {
	binary.LittleEndian.PutUint64(*b, v)
	*b = (*b)[8:]
}
