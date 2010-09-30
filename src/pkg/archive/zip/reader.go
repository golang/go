// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
The zip package provides support for reading ZIP archives.

See: http://www.pkware.com/documents/casestudies/APPNOTE.TXT

This package does not support ZIP64 or disk spanning.
*/
package zip

import (
	"bufio"
	"bytes"
	"compress/flate"
	"hash"
	"hash/crc32"
	"encoding/binary"
	"io"
	"os"
)

var (
	FormatError       = os.NewError("not a valid zip file")
	UnsupportedMethod = os.NewError("unsupported compression algorithm")
	ChecksumError     = os.NewError("checksum error")
)

type Reader struct {
	r       io.ReaderAt
	File    []*File
	Comment string
}

type File struct {
	FileHeader
	zipr         io.ReaderAt
	zipsize      int64
	headerOffset uint32
	bodyOffset   int64
}

// OpenReader will open the Zip file specified by name and return a Reader.
func OpenReader(name string) (*Reader, os.Error) {
	f, err := os.Open(name, os.O_RDONLY, 0644)
	if err != nil {
		return nil, err
	}
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	return NewReader(f, fi.Size)
}

// NewReader returns a new Reader reading from r, which is assumed to
// have the given size in bytes.
func NewReader(r io.ReaderAt, size int64) (*Reader, os.Error) {
	end, err := readDirectoryEnd(r, size)
	if err != nil {
		return nil, err
	}
	z := &Reader{
		r:       r,
		File:    make([]*File, end.directoryRecords),
		Comment: end.comment,
	}
	rs := io.NewSectionReader(r, 0, size)
	if _, err = rs.Seek(int64(end.directoryOffset), 0); err != nil {
		return nil, err
	}
	buf := bufio.NewReader(rs)
	for i := range z.File {
		z.File[i] = &File{zipr: r, zipsize: size}
		if err := readDirectoryHeader(z.File[i], buf); err != nil {
			return nil, err
		}
	}
	return z, nil
}

// Open returns a ReadCloser that provides access to the File's contents.
func (f *File) Open() (rc io.ReadCloser, err os.Error) {
	off := int64(f.headerOffset)
	if f.bodyOffset == 0 {
		r := io.NewSectionReader(f.zipr, off, f.zipsize-off)
		if err = readFileHeader(f, r); err != nil {
			return
		}
		if f.bodyOffset, err = r.Seek(0, 1); err != nil {
			return
		}
	}
	r := io.NewSectionReader(f.zipr, off+f.bodyOffset, int64(f.CompressedSize))
	switch f.Method {
	case 0: // store (no compression)
		rc = nopCloser{r}
	case 8: // DEFLATE
		rc = flate.NewReader(r)
	default:
		err = UnsupportedMethod
	}
	if rc != nil {
		rc = &checksumReader{rc, crc32.NewIEEE(), f.CRC32}
	}
	return
}

type checksumReader struct {
	rc   io.ReadCloser
	hash hash.Hash32
	sum  uint32
}

func (r *checksumReader) Read(b []byte) (n int, err os.Error) {
	n, err = r.rc.Read(b)
	r.hash.Write(b[:n])
	if err != os.EOF {
		return
	}
	if r.hash.Sum32() != r.sum {
		err = ChecksumError
	}
	return
}

func (r *checksumReader) Close() os.Error { return r.rc.Close() }

type nopCloser struct {
	io.Reader
}

func (f nopCloser) Close() os.Error { return nil }

func readFileHeader(f *File, r io.Reader) (err os.Error) {
	defer func() {
		if rerr, ok := recover().(os.Error); ok {
			err = rerr
		}
	}()
	var (
		signature      uint32
		filenameLength uint16
		extraLength    uint16
	)
	read(r, &signature)
	if signature != fileHeaderSignature {
		return FormatError
	}
	read(r, &f.ReaderVersion)
	read(r, &f.Flags)
	read(r, &f.Method)
	read(r, &f.ModifiedTime)
	read(r, &f.ModifiedDate)
	read(r, &f.CRC32)
	read(r, &f.CompressedSize)
	read(r, &f.UncompressedSize)
	read(r, &filenameLength)
	read(r, &extraLength)
	f.Name = string(readByteSlice(r, filenameLength))
	f.Extra = readByteSlice(r, extraLength)
	return
}

func readDirectoryHeader(f *File, r io.Reader) (err os.Error) {
	defer func() {
		if rerr, ok := recover().(os.Error); ok {
			err = rerr
		}
	}()
	var (
		signature          uint32
		filenameLength     uint16
		extraLength        uint16
		commentLength      uint16
		startDiskNumber    uint16 // unused
		internalAttributes uint16 // unused
		externalAttributes uint32 // unused
	)
	read(r, &signature)
	if signature != directoryHeaderSignature {
		return FormatError
	}
	read(r, &f.CreatorVersion)
	read(r, &f.ReaderVersion)
	read(r, &f.Flags)
	read(r, &f.Method)
	read(r, &f.ModifiedTime)
	read(r, &f.ModifiedDate)
	read(r, &f.CRC32)
	read(r, &f.CompressedSize)
	read(r, &f.UncompressedSize)
	read(r, &filenameLength)
	read(r, &extraLength)
	read(r, &commentLength)
	read(r, &startDiskNumber)
	read(r, &internalAttributes)
	read(r, &externalAttributes)
	read(r, &f.headerOffset)
	f.Name = string(readByteSlice(r, filenameLength))
	f.Extra = readByteSlice(r, extraLength)
	f.Comment = string(readByteSlice(r, commentLength))
	return
}

func readDirectoryEnd(r io.ReaderAt, size int64) (d *directoryEnd, err os.Error) {
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
	defer func() {
		if rerr, ok := recover().(os.Error); ok {
			err = rerr
			d = nil
		}
	}()
	br := bytes.NewBuffer(b[4:]) // skip over signature
	d = new(directoryEnd)
	read(br, &d.diskNbr)
	read(br, &d.dirDiskNbr)
	read(br, &d.dirRecordsThisDisk)
	read(br, &d.directoryRecords)
	read(br, &d.directorySize)
	read(br, &d.directoryOffset)
	read(br, &d.commentLen)
	d.comment = string(readByteSlice(br, d.commentLen))
	return d, nil
}

func findSignatureInBlock(b []byte) int {
	const minSize = 4 + 2 + 2 + 2 + 2 + 4 + 4 + 2 // fixed part of header
	for i := len(b) - minSize; i >= 0; i-- {
		// defined from directoryEndSignature in struct.go
		if b[i] == 'P' && b[i+1] == 'K' && b[i+2] == 0x05 && b[i+3] == 0x06 {
			// n is length of comment
			n := int(b[i+minSize-2]) | int(b[i+minSize-1])<<8
			if n+minSize+i == len(b) {
				return i
			}
		}
	}
	return -1
}

func read(r io.Reader, data interface{}) {
	if err := binary.Read(r, binary.LittleEndian, data); err != nil {
		panic(err)
	}
}

func readByteSlice(r io.Reader, l uint16) []byte {
	b := make([]byte, l)
	if l == 0 {
		return b
	}
	if _, err := io.ReadFull(r, b); err != nil {
		panic(err)
	}
	return b
}
