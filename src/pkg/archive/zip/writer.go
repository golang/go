// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zip

import (
	"bufio"
	"compress/flate"
	"encoding/binary"
	"errors"
	"hash"
	"hash/crc32"
	"io"
)

// TODO(adg): support zip file comments
// TODO(adg): support specifying deflate level

// Writer implements a zip file writer.
type Writer struct {
	countWriter
	dir    []*header
	last   *fileWriter
	closed bool
}

type header struct {
	*FileHeader
	offset uint32
}

// NewWriter returns a new Writer writing a zip file to w.
func NewWriter(w io.Writer) *Writer {
	return &Writer{countWriter: countWriter{w: bufio.NewWriter(w)}}
}

// Close finishes writing the zip file by writing the central directory.
// It does not (and can not) close the underlying writer.
func (w *Writer) Close() (err error) {
	if w.last != nil && !w.last.closed {
		if err = w.last.close(); err != nil {
			return
		}
		w.last = nil
	}
	if w.closed {
		return errors.New("zip: writer closed twice")
	}
	w.closed = true

	defer recoverError(&err)

	// write central directory
	start := w.count
	for _, h := range w.dir {
		write(w, uint32(directoryHeaderSignature))
		write(w, h.CreatorVersion)
		write(w, h.ReaderVersion)
		write(w, h.Flags)
		write(w, h.Method)
		write(w, h.ModifiedTime)
		write(w, h.ModifiedDate)
		write(w, h.CRC32)
		write(w, h.CompressedSize)
		write(w, h.UncompressedSize)
		write(w, uint16(len(h.Name)))
		write(w, uint16(len(h.Extra)))
		write(w, uint16(len(h.Comment)))
		write(w, uint16(0)) // disk number start
		write(w, uint16(0)) // internal file attributes
		write(w, h.ExternalAttrs)
		write(w, h.offset)
		writeBytes(w, []byte(h.Name))
		writeBytes(w, h.Extra)
		writeBytes(w, []byte(h.Comment))
	}
	end := w.count

	// write end record
	write(w, uint32(directoryEndSignature))
	write(w, uint16(0))          // disk number
	write(w, uint16(0))          // disk number where directory starts
	write(w, uint16(len(w.dir))) // number of entries this disk
	write(w, uint16(len(w.dir))) // number of entries total
	write(w, uint32(end-start))  // size of directory
	write(w, uint32(start))      // start of directory
	write(w, uint16(0))          // size of comment

	return w.w.(*bufio.Writer).Flush()
}

// Create adds a file to the zip file using the provided name.
// It returns a Writer to which the file contents should be written.
// The file's contents must be written to the io.Writer before the next
// call to Create, CreateHeader, or Close.
func (w *Writer) Create(name string) (io.Writer, error) {
	header := &FileHeader{
		Name:   name,
		Method: Deflate,
	}
	return w.CreateHeader(header)
}

// CreateHeader adds a file to the zip file using the provided FileHeader
// for the file metadata. 
// It returns a Writer to which the file contents should be written.
// The file's contents must be written to the io.Writer before the next
// call to Create, CreateHeader, or Close.
func (w *Writer) CreateHeader(fh *FileHeader) (io.Writer, error) {
	if w.last != nil && !w.last.closed {
		if err := w.last.close(); err != nil {
			return nil, err
		}
	}

	fh.Flags |= 0x8 // we will write a data descriptor
	fh.CreatorVersion = fh.CreatorVersion&0xff00 | 0x14
	fh.ReaderVersion = 0x14

	fw := &fileWriter{
		zipw:      w,
		compCount: &countWriter{w: w},
		crc32:     crc32.NewIEEE(),
	}
	switch fh.Method {
	case Store:
		fw.comp = nopCloser{fw.compCount}
	case Deflate:
		fw.comp = flate.NewWriter(fw.compCount, 5)
	default:
		return nil, ErrAlgorithm
	}
	fw.rawCount = &countWriter{w: fw.comp}

	h := &header{
		FileHeader: fh,
		offset:     uint32(w.count),
	}
	w.dir = append(w.dir, h)
	fw.header = h

	if err := writeHeader(w, fh); err != nil {
		return nil, err
	}

	w.last = fw
	return fw, nil
}

func writeHeader(w io.Writer, h *FileHeader) (err error) {
	defer recoverError(&err)
	write(w, uint32(fileHeaderSignature))
	write(w, h.ReaderVersion)
	write(w, h.Flags)
	write(w, h.Method)
	write(w, h.ModifiedTime)
	write(w, h.ModifiedDate)
	write(w, h.CRC32)
	write(w, h.CompressedSize)
	write(w, h.UncompressedSize)
	write(w, uint16(len(h.Name)))
	write(w, uint16(len(h.Extra)))
	writeBytes(w, []byte(h.Name))
	writeBytes(w, h.Extra)
	return nil
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

func (w *fileWriter) close() (err error) {
	if w.closed {
		return errors.New("zip: file closed twice")
	}
	w.closed = true
	if err = w.comp.Close(); err != nil {
		return
	}

	// update FileHeader
	fh := w.header.FileHeader
	fh.CRC32 = w.crc32.Sum32()
	fh.CompressedSize = uint32(w.compCount.count)
	fh.UncompressedSize = uint32(w.rawCount.count)

	// write data descriptor
	defer recoverError(&err)
	write(w.zipw, fh.CRC32)
	write(w.zipw, fh.CompressedSize)
	write(w.zipw, fh.UncompressedSize)

	return nil
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

func write(w io.Writer, data interface{}) {
	if err := binary.Write(w, binary.LittleEndian, data); err != nil {
		panic(err)
	}
}

func writeBytes(w io.Writer, b []byte) {
	n, err := w.Write(b)
	if err != nil {
		panic(err)
	}
	if n != len(b) {
		panic(io.ErrShortWrite)
	}
}
