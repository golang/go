// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multipart

import (
	"bytes"
	"fmt"
	"io"
	"net/textproto"
	"os"
	"rand"
	"strings"
)

// Writer is used to generate multipart messages.
type Writer struct {
	// Boundary is the random boundary string between
	// parts. NewWriter will generate this but it must
	// not be changed after a part has been created.
	// Setting this to an invalid value will generate
	// malformed messages.
	Boundary string

	w        io.Writer
	lastpart *part
}

// NewWriter returns a new multipart Writer with a random boundary,
// writing to w.
func NewWriter(w io.Writer) *Writer {
	return &Writer{
		w:        w,
		Boundary: randomBoundary(),
	}
}

// FormDataContentType returns the Content-Type for an HTTP
// multipart/form-data with this Writer's Boundary.
func (w *Writer) FormDataContentType() string {
	return "multipart/form-data; boundary=" + w.Boundary
}

const randChars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

func randomBoundary() string {
	var buf [60]byte
	for i := range buf {
		buf[i] = randChars[rand.Intn(len(randChars))]
	}
	return string(buf[:])
}

// CreatePart creates a new multipart section with the provided
// header. The previous part, if still open, is closed. The body of
// the part should be written to the returned WriteCloser. Closing the
// returned WriteCloser after writing is optional.
func (w *Writer) CreatePart(header textproto.MIMEHeader) (io.WriteCloser, os.Error) {
	if w.lastpart != nil {
		if err := w.lastpart.Close(); err != nil {
			return nil, err
		}
	}
	var b bytes.Buffer
	fmt.Fprintf(&b, "\r\n--%s\r\n", w.Boundary)
	// TODO(bradfitz): move this to textproto.MimeHeader.Write(w), have it sort
	// and clean, like http.Header.Write(w) does.
	for k, vv := range header {
		for _, v := range vv {
			fmt.Fprintf(&b, "%s: %s\r\n", k, v)
		}
	}
	fmt.Fprintf(&b, "\r\n")
	_, err := io.Copy(w.w, &b)
	if err != nil {
		return nil, err
	}
	p := &part{
		mw: w,
	}
	w.lastpart = p
	return p, nil
}

func escapeQuotes(s string) string {
	s = strings.Replace(s, "\\", "\\\\", -1)
	s = strings.Replace(s, "\"", "\\\"", -1)
	return s
}

// CreateFormFile is a convenience wrapper around CreatePart. It creates
// a new form-data header with the provided field name and file name.
func (w *Writer) CreateFormFile(fieldname, filename string) (io.WriteCloser, os.Error) {
	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition",
		fmt.Sprintf(`form-data; name="%s"; filename="%s"`,
			escapeQuotes(fieldname), escapeQuotes(filename)))
	h.Set("Content-Type", "application/octet-stream")
	return w.CreatePart(h)
}

// CreateFormField is a convenience wrapper around CreatePart. It creates
// a new form-data header with the provided field name.
func (w *Writer) CreateFormField(fieldname string) (io.WriteCloser, os.Error) {
	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition",
		fmt.Sprintf(`form-data; name="%s"`, escapeQuotes(fieldname)))
	return w.CreatePart(h)
}

// WriteField is a convenience wrapper around CreateFormField. It creates and
// writes a part with the provided name and value.
func (w *Writer) WriteField(fieldname, value string) os.Error {
	p, err := w.CreateFormField(fieldname)
	if err != nil {
		return err
	}
	_, err = p.Write([]byte(value))
	if err != nil {
		return err
	}
	return p.Close()
}

// Close finishes the multipart message. It closes the previous part,
// if still open, and writes the trailing boundary end line to the
// output.
func (w *Writer) Close() os.Error {
	if w.lastpart != nil {
		if err := w.lastpart.Close(); err != nil {
			return err
		}
		w.lastpart = nil
	}
	_, err := fmt.Fprintf(w.w, "\r\n--%s--\r\n", w.Boundary)
	return err
}

type part struct {
	mw     *Writer
	closed bool
	we     os.Error // last error that occurred writing
}

func (p *part) Close() os.Error {
	p.closed = true
	return p.we
}

func (p *part) Write(d []byte) (n int, err os.Error) {
	if p.closed {
		return 0, os.NewError("multipart: Write after Close")
	}
	n, err = p.mw.w.Write(d)
	if err != nil {
		p.we = err
	}
	return
}
