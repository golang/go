// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zip

import (
	"bufio"
	"compress/flate"
	"hash"
	"hash/crc32"
	"encoding/binary"
	"io"
	"io/ioutil"
	"os"
)

var (
	FormatError       = os.NewError("zip: not a valid zip file")
	UnsupportedMethod = os.NewError("zip: unsupported compression algorithm")
	ChecksumError     = os.NewError("zip: checksum error")
)

type Reader struct {
	r       io.ReaderAt
	File    []*File
	Comment string
}

type ReadCloser struct {
	f *os.File
	Reader
}

type File struct {
	FileHeader
	zipr         io.ReaderAt
	zipsize      int64
	headerOffset int64
}

func (f *File) hasDataDescriptor() bool {
	return f.Flags&0x8 != 0
}

// OpenReader will open the Zip file specified by name and return a ReadCloser.
func OpenReader(name string) (*ReadCloser, os.Error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	r := new(ReadCloser)
	if err := r.init(f, fi.Size); err != nil {
		f.Close()
		return nil, err
	}
	return r, nil
}

// NewReader returns a new Reader reading from r, which is assumed to
// have the given size in bytes.
func NewReader(r io.ReaderAt, size int64) (*Reader, os.Error) {
	zr := new(Reader)
	if err := zr.init(r, size); err != nil {
		return nil, err
	}
	return zr, nil
}

func (z *Reader) init(r io.ReaderAt, size int64) os.Error {
	end, err := readDirectoryEnd(r, size)
	if err != nil {
		return err
	}
	z.r = r
	z.File = make([]*File, 0, end.directoryRecords)
	z.Comment = end.comment
	rs := io.NewSectionReader(r, 0, size)
	if _, err = rs.Seek(int64(end.directoryOffset), os.SEEK_SET); err != nil {
		return err
	}
	buf := bufio.NewReader(rs)

	// The count of files inside a zip is truncated to fit in a uint16.
	// Gloss over this by reading headers until we encounter
	// a bad one, and then only report a FormatError or UnexpectedEOF if
	// the file count modulo 65536 is incorrect.
	for {
		f := &File{zipr: r, zipsize: size}
		err = readDirectoryHeader(f, buf)
		if err == FormatError || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			return err
		}
		z.File = append(z.File, f)
	}
	if uint16(len(z.File)) != end.directoryRecords {
		// Return the readDirectoryHeader error if we read
		// the wrong number of directory entries.
		return err
	}
	return nil
}

// Close closes the Zip file, rendering it unusable for I/O.
func (rc *ReadCloser) Close() os.Error {
	return rc.f.Close()
}

// Open returns a ReadCloser that provides access to the File's contents.
// It is safe to Open and Read from files concurrently.
func (f *File) Open() (rc io.ReadCloser, err os.Error) {
	bodyOffset, err := f.findBodyOffset()
	if err != nil {
		return
	}
	size := int64(f.CompressedSize)
	if size == 0 && f.hasDataDescriptor() {
		// permit SectionReader to see the rest of the file
		size = f.zipsize - (f.headerOffset + bodyOffset)
	}
	r := io.NewSectionReader(f.zipr, f.headerOffset+bodyOffset, size)
	switch f.Method {
	case Store: // (no compression)
		rc = ioutil.NopCloser(r)
	case Deflate:
		rc = flate.NewReader(r)
	default:
		err = UnsupportedMethod
	}
	if rc != nil {
		rc = &checksumReader{rc, crc32.NewIEEE(), f, r}
	}
	return
}

type checksumReader struct {
	rc   io.ReadCloser
	hash hash.Hash32
	f    *File
	zipr io.Reader // for reading the data descriptor
}

func (r *checksumReader) Read(b []byte) (n int, err os.Error) {
	n, err = r.rc.Read(b)
	r.hash.Write(b[:n])
	if err != os.EOF {
		return
	}
	if r.f.hasDataDescriptor() {
		if err = readDataDescriptor(r.zipr, r.f); err != nil {
			return
		}
	}
	if r.hash.Sum32() != r.f.CRC32 {
		err = ChecksumError
	}
	return
}

func (r *checksumReader) Close() os.Error { return r.rc.Close() }

func readFileHeader(f *File, r io.Reader) os.Error {
	var b [fileHeaderLen]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return err
	}
	c := binary.LittleEndian
	if sig := c.Uint32(b[:4]); sig != fileHeaderSignature {
		return FormatError
	}
	f.ReaderVersion = c.Uint16(b[4:6])
	f.Flags = c.Uint16(b[6:8])
	f.Method = c.Uint16(b[8:10])
	f.ModifiedTime = c.Uint16(b[10:12])
	f.ModifiedDate = c.Uint16(b[12:14])
	f.CRC32 = c.Uint32(b[14:18])
	f.CompressedSize = c.Uint32(b[18:22])
	f.UncompressedSize = c.Uint32(b[22:26])
	filenameLen := int(c.Uint16(b[26:28]))
	extraLen := int(c.Uint16(b[28:30]))
	d := make([]byte, filenameLen+extraLen)
	if _, err := io.ReadFull(r, d); err != nil {
		return err
	}
	f.Name = string(d[:filenameLen])
	f.Extra = d[filenameLen:]
	return nil
}

// findBodyOffset does the minimum work to verify the file has a header
// and returns the file body offset.
func (f *File) findBodyOffset() (int64, os.Error) {
	r := io.NewSectionReader(f.zipr, f.headerOffset, f.zipsize-f.headerOffset)
	var b [fileHeaderLen]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	c := binary.LittleEndian
	if sig := c.Uint32(b[:4]); sig != fileHeaderSignature {
		return 0, FormatError
	}
	filenameLen := int(c.Uint16(b[26:28]))
	extraLen := int(c.Uint16(b[28:30]))
	return int64(fileHeaderLen + filenameLen + extraLen), nil
}

// readDirectoryHeader attempts to read a directory header from r.
// It returns io.ErrUnexpectedEOF if it cannot read a complete header,
// and FormatError if it doesn't find a valid header signature.
func readDirectoryHeader(f *File, r io.Reader) os.Error {
	var b [directoryHeaderLen]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return err
	}
	c := binary.LittleEndian
	if sig := c.Uint32(b[:4]); sig != directoryHeaderSignature {
		return FormatError
	}
	f.CreatorVersion = c.Uint16(b[4:6])
	f.ReaderVersion = c.Uint16(b[6:8])
	f.Flags = c.Uint16(b[8:10])
	f.Method = c.Uint16(b[10:12])
	f.ModifiedTime = c.Uint16(b[12:14])
	f.ModifiedDate = c.Uint16(b[14:16])
	f.CRC32 = c.Uint32(b[16:20])
	f.CompressedSize = c.Uint32(b[20:24])
	f.UncompressedSize = c.Uint32(b[24:28])
	filenameLen := int(c.Uint16(b[28:30]))
	extraLen := int(c.Uint16(b[30:32]))
	commentLen := int(c.Uint16(b[32:34]))
	// startDiskNumber := c.Uint16(b[34:36])    // Unused
	// internalAttributes := c.Uint16(b[36:38]) // Unused
	f.ExternalAttrs = c.Uint32(b[38:42])
	f.headerOffset = int64(c.Uint32(b[42:46]))
	d := make([]byte, filenameLen+extraLen+commentLen)
	if _, err := io.ReadFull(r, d); err != nil {
		return err
	}
	f.Name = string(d[:filenameLen])
	f.Extra = d[filenameLen : filenameLen+extraLen]
	f.Comment = string(d[filenameLen+extraLen:])
	return nil
}

func readDataDescriptor(r io.Reader, f *File) os.Error {
	var b [dataDescriptorLen]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return err
	}
	c := binary.LittleEndian
	f.CRC32 = c.Uint32(b[:4])
	f.CompressedSize = c.Uint32(b[4:8])
	f.UncompressedSize = c.Uint32(b[8:12])
	return nil
}

func readDirectoryEnd(r io.ReaderAt, size int64) (dir *directoryEnd, err os.Error) {
	// look for directoryEndSignature in the last 1k, then in the last 65k
	var b []byte
	for i, bLen := range []int64{1024, 65 * 1024} {
		if bLen > size {
			bLen = size
		}
		b = make([]byte, int(bLen))
		if _, err := r.ReadAt(b, size-bLen); err != nil && err != os.EOF {
			return nil, err
		}
		if p := findSignatureInBlock(b); p >= 0 {
			b = b[p:]
			break
		}
		if i == 1 || bLen == size {
			return nil, FormatError
		}
	}

	// read header into struct
	c := binary.LittleEndian
	d := new(directoryEnd)
	d.diskNbr = c.Uint16(b[4:6])
	d.dirDiskNbr = c.Uint16(b[6:8])
	d.dirRecordsThisDisk = c.Uint16(b[8:10])
	d.directoryRecords = c.Uint16(b[10:12])
	d.directorySize = c.Uint32(b[12:16])
	d.directoryOffset = c.Uint32(b[16:20])
	d.commentLen = c.Uint16(b[20:22])
	d.comment = string(b[22 : 22+int(d.commentLen)])
	return d, nil
}

func findSignatureInBlock(b []byte) int {
	for i := len(b) - directoryEndLen; i >= 0; i-- {
		// defined from directoryEndSignature in struct.go
		if b[i] == 'P' && b[i+1] == 'K' && b[i+2] == 0x05 && b[i+3] == 0x06 {
			// n is length of comment
			n := int(b[i+directoryEndLen-2]) | int(b[i+directoryEndLen-1])<<8
			if n+directoryEndLen+i == len(b) {
				return i
			}
		}
	}
	return -1
}
