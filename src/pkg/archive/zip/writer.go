// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zip

import (
	"bufio"
	"compress/flate"
	"errors"
	"hash"
	"hash/crc32"
	"io"
)

// TODO(adg): support zip file comments
// TODO(adg): support specifying deflate level

// Writer implements a zip file writer.
type Writer struct {
	cw     *countWriter
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
	return &Writer{cw: &countWriter{w: bufio.NewWriter(w)}}
}

// Close finishes writing the zip file by writing the central directory.
// It does not (and can not) close the underlying writer.
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
		var b [directoryHeaderLen]byte
		putUint32(b[:], uint32(directoryHeaderSignature))
		putUint16(b[4:], h.CreatorVersion)
		putUint16(b[6:], h.ReaderVersion)
		putUint16(b[8:], h.Flags)
		putUint16(b[10:], h.Method)
		putUint16(b[12:], h.ModifiedTime)
		putUint16(b[14:], h.ModifiedDate)
		putUint32(b[16:], h.CRC32)
		putUint32(b[20:], h.CompressedSize)
		putUint32(b[24:], h.UncompressedSize)
		putUint16(b[28:], uint16(len(h.Name)))
		putUint16(b[30:], uint16(len(h.Extra)))
		putUint16(b[32:], uint16(len(h.Comment)))
		// skip two uint16's, disk number start and internal file attributes
		putUint32(b[38:], h.ExternalAttrs)
		putUint32(b[42:], h.offset)
		if _, err := w.cw.Write(b[:]); err != nil {
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

	// write end record
	var b [directoryEndLen]byte
	putUint32(b[:], uint32(directoryEndSignature))
	putUint16(b[4:], uint16(0))           // disk number
	putUint16(b[6:], uint16(0))           // disk number where directory starts
	putUint16(b[8:], uint16(len(w.dir)))  // number of entries this disk
	putUint16(b[10:], uint16(len(w.dir))) // number of entries total
	putUint32(b[12:], uint32(end-start))  // size of directory
	putUint32(b[16:], uint32(start))      // start of directory
	// skipped size of comment (always zero)
	if _, err := w.cw.Write(b[:]); err != nil {
		return err
	}

	return w.cw.w.(*bufio.Writer).Flush()
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
		zipw:      w.cw,
		compCount: &countWriter{w: w.cw},
		crc32:     crc32.NewIEEE(),
	}
	switch fh.Method {
	case Store:
		fw.comp = nopCloser{fw.compCount}
	case Deflate:
		var err error
		fw.comp, err = flate.NewWriter(fw.compCount, 5)
		if err != nil {
			return nil, err
		}
	default:
		return nil, ErrAlgorithm
	}
	fw.rawCount = &countWriter{w: fw.comp}

	h := &header{
		FileHeader: fh,
		offset:     uint32(w.cw.count),
	}
	w.dir = append(w.dir, h)
	fw.header = h

	if err := writeHeader(w.cw, fh); err != nil {
		return nil, err
	}

	w.last = fw
	return fw, nil
}

func writeHeader(w io.Writer, h *FileHeader) error {
	var b [fileHeaderLen]byte
	putUint32(b[:], uint32(fileHeaderSignature))
	putUint16(b[4:], h.ReaderVersion)
	putUint16(b[6:], h.Flags)
	putUint16(b[8:], h.Method)
	putUint16(b[10:], h.ModifiedTime)
	putUint16(b[12:], h.ModifiedDate)
	putUint32(b[14:], h.CRC32)
	putUint32(b[18:], h.CompressedSize)
	putUint32(b[22:], h.UncompressedSize)
	putUint16(b[26:], uint16(len(h.Name)))
	putUint16(b[28:], uint16(len(h.Extra)))
	if _, err := w.Write(b[:]); err != nil {
		return err
	}
	if _, err := io.WriteString(w, h.Name); err != nil {
		return err
	}
	_, err := w.Write(h.Extra)
	return err
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
	fh.CompressedSize = uint32(w.compCount.count)
	fh.UncompressedSize = uint32(w.rawCount.count)

	// write data descriptor
	var b [dataDescriptorLen]byte
	putUint32(b[:], fh.CRC32)
	putUint32(b[4:], fh.CompressedSize)
	putUint32(b[8:], fh.UncompressedSize)
	_, err := w.zipw.Write(b[:])
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

// We use these putUintXX functions instead of encoding/binary's Write to avoid
// reflection. It's easy enough, anyway.

func putUint16(b []byte, v uint16) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
}

func putUint32(b []byte, v uint32) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
}
