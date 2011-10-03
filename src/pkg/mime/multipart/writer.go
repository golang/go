// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multipart

import (
	"bytes"
	"crypto/rand"
	"fmt"
	"io"
	"net/textproto"
	"os"
	"strings"
)

// A Writer generates multipart messages.
type Writer struct {
	w        io.Writer
	boundary string
	lastpart *part
}

// NewWriter returns a new multipart Writer with a random boundary,
// writing to w.
func NewWriter(w io.Writer) *Writer {
	return &Writer{
		w:        w,
		boundary: randomBoundary(),
	}
}

// Boundary returns the Writer's randomly selected boundary string.
func (w *Writer) Boundary() string {
	return w.boundary
}

// FormDataContentType returns the Content-Type for an HTTP
// multipart/form-data with this Writer's Boundary.
func (w *Writer) FormDataContentType() string {
	return "multipart/form-data; boundary=" + w.boundary
}

func randomBoundary() string {
	var buf [30]byte
	_, err := io.ReadFull(rand.Reader, buf[:])
	if err != nil {
		panic(err)
	}
	return fmt.Sprintf("%x", buf[:])
}

// CreatePart creates a new multipart section with the provided
// header. The body of the part should be written to the returned
// Writer. After calling CreatePart, any previous part may no longer
// be written to.
func (w *Writer) CreatePart(header textproto.MIMEHeader) (io.Writer, os.Error) {
	if w.lastpart != nil {
		if err := w.lastpart.close(); err != nil {
			return nil, err
		}
	}
	var b bytes.Buffer
	if w.lastpart != nil {
		fmt.Fprintf(&b, "\r\n--%s\r\n", w.boundary)
	} else {
		fmt.Fprintf(&b, "--%s\r\n", w.boundary)
	}
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

var quoteEscaper = strings.NewReplacer("\\", "\\\\", `"`, "\\\"")

func escapeQuotes(s string) string {
	return quoteEscaper.Replace(s)
}

// CreateFormFile is a convenience wrapper around CreatePart. It creates
// a new form-data header with the provided field name and file name.
func (w *Writer) CreateFormFile(fieldname, filename string) (io.Writer, os.Error) {
	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition",
		fmt.Sprintf(`form-data; name="%s"; filename="%s"`,
			escapeQuotes(fieldname), escapeQuotes(filename)))
	h.Set("Content-Type", "application/octet-stream")
	return w.CreatePart(h)
}

// CreateFormField calls CreatePart with a header using the
// given field name.
func (w *Writer) CreateFormField(fieldname string) (io.Writer, os.Error) {
	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition",
		fmt.Sprintf(`form-data; name="%s"`, escapeQuotes(fieldname)))
	return w.CreatePart(h)
}

// WriteField calls CreateFormField and then writes the given value.
func (w *Writer) WriteField(fieldname, value string) os.Error {
	p, err := w.CreateFormField(fieldname)
	if err != nil {
		return err
	}
	_, err = p.Write([]byte(value))
	return err
}

// Close finishes the multipart message and writes the trailing
// boundary end line to the output.
func (w *Writer) Close() os.Error {
	if w.lastpart != nil {
		if err := w.lastpart.close(); err != nil {
			return err
		}
		w.lastpart = nil
	}
	_, err := fmt.Fprintf(w.w, "\r\n--%s--\r\n", w.boundary)
	return err
}

type part struct {
	mw     *Writer
	closed bool
	we     os.Error // last error that occurred writing
}

func (p *part) close() os.Error {
	p.closed = true
	return p.we
}

func (p *part) Write(d []byte) (n int, err os.Error) {
	if p.closed {
		return 0, os.NewError("multipart: can't write to finished part")
	}
	n, err = p.mw.w.Write(d)
	if err != nil {
		p.we = err
	}
	return
}
