// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multipart

import (
	"bytes"
	"errors"
	"io"
	"io/ioutil"
	"net/textproto"
	"os"
)

// TODO(adg,bradfitz): find a way to unify the DoS-prevention strategy here
// with that of the http package's ParseForm.

// ReadForm parses an entire multipart message whose parts have
// a Content-Disposition of "form-data".
// It stores up to maxMemory bytes of the file parts in memory
// and the remainder on disk in temporary files.
func (r *Reader) ReadForm(maxMemory int64) (f *Form, err error) {
	form := &Form{make(map[string][]string), make(map[string][]*FileHeader)}
	defer func() {
		if err != nil {
			form.RemoveAll()
		}
	}()

	maxValueBytes := int64(10 << 20) // 10 MB is a lot of text.
	for {
		p, err := r.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		name := p.FormName()
		if name == "" {
			continue
		}
		filename := p.FileName()

		var b bytes.Buffer

		if filename == "" {
			// value, store as string in memory
			n, err := io.CopyN(&b, p, maxValueBytes)
			if err != nil && err != io.EOF {
				return nil, err
			}
			maxValueBytes -= n
			if maxValueBytes == 0 {
				return nil, errors.New("multipart: message too large")
			}
			form.Value[name] = append(form.Value[name], b.String())
			continue
		}

		// file, store in memory or on disk
		fh := &FileHeader{
			Filename: filename,
			Header:   p.Header,
		}
		n, err := io.CopyN(&b, p, maxMemory+1)
		if err != nil && err != io.EOF {
			return nil, err
		}
		if n > maxMemory {
			// too big, write to disk and flush buffer
			file, err := ioutil.TempFile("", "multipart-")
			if err != nil {
				return nil, err
			}
			defer file.Close()
			_, err = io.Copy(file, io.MultiReader(&b, p))
			if err != nil {
				os.Remove(file.Name())
				return nil, err
			}
			fh.tmpfile = file.Name()
		} else {
			fh.content = b.Bytes()
			maxMemory -= n
		}
		form.File[name] = append(form.File[name], fh)
	}

	return form, nil
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
				if e != nil && err == nil {
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

	content []byte
	tmpfile string
}

// Open opens and returns the FileHeader's associated File.
func (fh *FileHeader) Open() (File, error) {
	if b := fh.content; b != nil {
		r := io.NewSectionReader(sliceReaderAt(b), 0, int64(len(b)))
		return sectionReadCloser{r}, nil
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
}

func (rc sectionReadCloser) Close() error {
	return nil
}

type sliceReaderAt []byte

func (r sliceReaderAt) ReadAt(b []byte, off int64) (int, error) {
	if int(off) >= len(r) || off < 0 {
		return 0, io.ErrUnexpectedEOF
	}
	n := copy(b, r[int(off):])
	return n, nil
}
