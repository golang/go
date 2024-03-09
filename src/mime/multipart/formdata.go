// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multipart

import (
	"bytes"
	"errors"
	"internal/godebug"
	"io"
	"math"
	"net/textproto"
	"os"
	"strconv"
)

// ErrMessageTooLarge is returned by ReadForm if the message form
// data is too large to be processed.
var ErrMessageTooLarge = errors.New("multipart: message too large")

// TODO(adg,bradfitz): find a way to unify the DoS-prevention strategy here
// with that of the http package's ParseForm.

// ReadForm parses an entire multipart message whose parts have
// a Content-Disposition of "form-data".
// It stores up to maxMemory bytes + 10MB (reserved for non-file parts)
// in memory. File parts which can't be stored in memory will be stored on
// disk in temporary files.
// It returns ErrMessageTooLarge if all non-file parts can't be stored in
// memory.
func (r *Reader) ReadForm(maxMemory int64) (*Form, error) {
	return r.readForm(maxMemory)
}

var (
	multipartfiles    = godebug.New("#multipartfiles") // TODO: document and remove #
	multipartmaxparts = godebug.New("multipartmaxparts")
)

func (r *Reader) readForm(maxMemory int64) (_ *Form, err error) {
	form := &Form{make(map[string][]string), make(map[string][]*FileHeader)}
	var (
		file    *os.File
		fileOff int64
	)
	numDiskFiles := 0
	combineFiles := true
	if multipartfiles.Value() == "distinct" {
		combineFiles = false
		// multipartfiles.IncNonDefault() // TODO: uncomment after documenting
	}
	maxParts := 1000
	if s := multipartmaxparts.Value(); s != "" {
		if v, err := strconv.Atoi(s); err == nil && v >= 0 {
			maxParts = v
			multipartmaxparts.IncNonDefault()
		}
	}
	maxHeaders := maxMIMEHeaders()

	defer func() {
		if file != nil {
			if cerr := file.Close(); err == nil {
				err = cerr
			}
		}
		if combineFiles && numDiskFiles > 1 {
			for _, fhs := range form.File {
				for _, fh := range fhs {
					fh.tmpshared = true
				}
			}
		}
		if err != nil {
			form.RemoveAll()
			if file != nil {
				os.Remove(file.Name())
			}
		}
	}()

	// maxFileMemoryBytes is the maximum bytes of file data we will store in memory.
	// Data past this limit is written to disk.
	// This limit strictly applies to content, not metadata (filenames, MIME headers, etc.),
	// since metadata is always stored in memory, not disk.
	//
	// maxMemoryBytes is the maximum bytes we will store in memory, including file content,
	// non-file part values, metadata, and map entry overhead.
	//
	// We reserve an additional 10 MB in maxMemoryBytes for non-file data.
	//
	// The relationship between these parameters, as well as the overly-large and
	// unconfigurable 10 MB added on to maxMemory, is unfortunate but difficult to change
	// within the constraints of the API as documented.
	maxFileMemoryBytes := maxMemory
	if maxFileMemoryBytes == math.MaxInt64 {
		maxFileMemoryBytes--
	}
	maxMemoryBytes := maxMemory + int64(10<<20)
	if maxMemoryBytes <= 0 {
		if maxMemory < 0 {
			maxMemoryBytes = 0
		} else {
			maxMemoryBytes = math.MaxInt64
		}
	}
	var copyBuf []byte
	for {
		p, err := r.nextPart(false, maxMemoryBytes, maxHeaders)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if maxParts <= 0 {
			return nil, ErrMessageTooLarge
		}
		maxParts--

		name := p.FormName()
		if name == "" {
			continue
		}
		filename := p.FileName()

		// Multiple values for the same key (one map entry, longer slice) are cheaper
		// than the same number of values for different keys (many map entries), but
		// using a consistent per-value cost for overhead is simpler.
		const mapEntryOverhead = 200
		maxMemoryBytes -= int64(len(name))
		maxMemoryBytes -= mapEntryOverhead
		if maxMemoryBytes < 0 {
			// We can't actually take this path, since nextPart would already have
			// rejected the MIME headers for being too large. Check anyway.
			return nil, ErrMessageTooLarge
		}

		var b bytes.Buffer

		if filename == "" {
			// value, store as string in memory
			n, err := io.CopyN(&b, p, maxMemoryBytes+1)
			if err != nil && err != io.EOF {
				return nil, err
			}
			maxMemoryBytes -= n
			if maxMemoryBytes < 0 {
				return nil, ErrMessageTooLarge
			}
			form.Value[name] = append(form.Value[name], b.String())
			continue
		}

		// file, store in memory or on disk
		const fileHeaderSize = 100
		maxMemoryBytes -= mimeHeaderSize(p.Header)
		maxMemoryBytes -= mapEntryOverhead
		maxMemoryBytes -= fileHeaderSize
		if maxMemoryBytes < 0 {
			return nil, ErrMessageTooLarge
		}
		for _, v := range p.Header {
			maxHeaders -= int64(len(v))
		}
		fh := &FileHeader{
			Filename: filename,
			Header:   p.Header,
		}
		n, err := io.CopyN(&b, p, maxFileMemoryBytes+1)
		if err != nil && err != io.EOF {
			return nil, err
		}
		if n > maxFileMemoryBytes {
			if file == nil {
				file, err = os.CreateTemp(r.tempDir, "multipart-")
				if err != nil {
					return nil, err
				}
			}
			numDiskFiles++
			if _, err := file.Write(b.Bytes()); err != nil {
				return nil, err
			}
			if copyBuf == nil {
				copyBuf = make([]byte, 32*1024) // same buffer size as io.Copy uses
			}
			// os.File.ReadFrom will allocate its own copy buffer if we let io.Copy use it.
			type writerOnly struct{ io.Writer }
			remainingSize, err := io.CopyBuffer(writerOnly{file}, p, copyBuf)
			if err != nil {
				return nil, err
			}
			fh.tmpfile = file.Name()
			fh.Size = int64(b.Len()) + remainingSize
			fh.tmpoff = fileOff
			fileOff += fh.Size
			if !combineFiles {
				if err := file.Close(); err != nil {
					return nil, err
				}
				file = nil
			}
		} else {
			fh.content = b.Bytes()
			fh.Size = int64(len(fh.content))
			maxFileMemoryBytes -= n
			maxMemoryBytes -= n
		}
		form.File[name] = append(form.File[name], fh)
	}

	return form, nil
}

func mimeHeaderSize(h textproto.MIMEHeader) (size int64) {
	size = 400
	for k, vs := range h {
		size += int64(len(k))
		size += 200 // map entry overhead
		for _, v := range vs {
			size += int64(len(v))
		}
	}
	return size
}

// Form is a parsed multipart form.
// Its File parts are stored either in memory or on disk,
// and are accessible via the *FileHeader's Open method.
// Its Value parts are stored as strings.
// Both are keyed by field name.
type Form struct {
	Value map[string][]string
	File  map[string][]*FileHeader
}

// RemoveAll removes any temporary files associated with a Form.
func (f *Form) RemoveAll() error {
	var err error
	for _, fhs := range f.File {
		for _, fh := range fhs {
			if fh.tmpfile != "" {
				e := os.Remove(fh.tmpfile)
				if e != nil && !errors.Is(e, os.ErrNotExist) && err == nil {
					err = e
				}
			}
		}
	}
	return err
}

// A FileHeader describes a file part of a multipart request.
type FileHeader struct {
	Filename string
	Header   textproto.MIMEHeader
	Size     int64

	content   []byte
	tmpfile   string
	tmpoff    int64
	tmpshared bool
}

// Open opens and returns the FileHeader's associated File.
func (fh *FileHeader) Open() (File, error) {
	if b := fh.content; b != nil {
		r := io.NewSectionReader(bytes.NewReader(b), 0, int64(len(b)))
		return sectionReadCloser{r, nil}, nil
	}
	if fh.tmpshared {
		f, err := os.Open(fh.tmpfile)
		if err != nil {
			return nil, err
		}
		r := io.NewSectionReader(f, fh.tmpoff, fh.Size)
		return sectionReadCloser{r, f}, nil
	}
	return os.Open(fh.tmpfile)
}

// File is an interface to access the file part of a multipart message.
// Its contents may be either stored in memory or on disk.
// If stored on disk, the File's underlying concrete type will be an *os.File.
type File interface {
	io.Reader
	io.ReaderAt
	io.Seeker
	io.Closer
}

// helper types to turn a []byte into a File

type sectionReadCloser struct {
	*io.SectionReader
	io.Closer
}

func (rc sectionReadCloser) Close() error {
	if rc.Closer != nil {
		return rc.Closer.Close()
	}
	return nil
}
