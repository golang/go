// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestBodyReadBadTrailer(t *testing.T) {
	b := &body{
		src: strings.NewReader("foobar"),
		hdr: true, // force reading the trailer
		r:   bufio.NewReader(strings.NewReader("")),
	}
	buf := make([]byte, 7)
	n, err := b.Read(buf[:3])
	got := string(buf[:n])
	if got != "foo" || err != nil {
		t.Fatalf(`first Read = %d (%q), %v; want 3 ("foo")`, n, got, err)
	}

	n, err = b.Read(buf[:])
	got = string(buf[:n])
	if got != "bar" || err != nil {
		t.Fatalf(`second Read = %d (%q), %v; want 3 ("bar")`, n, got, err)
	}

	n, err = b.Read(buf[:])
	got = string(buf[:n])
	if err == nil {
		t.Errorf("final Read was successful (%q), expected error from trailer read", got)
	}
}

func TestFinalChunkedBodyReadEOF(t *testing.T) {
	res, err := ReadResponse(bufio.NewReader(strings.NewReader(
		"HTTP/1.1 200 OK\r\n"+
			"Transfer-Encoding: chunked\r\n"+
			"\r\n"+
			"0a\r\n"+
			"Body here\n\r\n"+
			"09\r\n"+
			"continued\r\n"+
			"0\r\n"+
			"\r\n")), nil)
	if err != nil {
		t.Fatal(err)
	}
	want := "Body here\ncontinued"
	buf := make([]byte, len(want))
	n, err := res.Body.Read(buf)
	if n != len(want) || err != io.EOF {
		t.Logf("body = %#v", res.Body)
		t.Errorf("Read = %v, %v; want %d, EOF", n, err, len(want))
	}
	if string(buf) != want {
		t.Errorf("buf = %q; want %q", buf, want)
	}
}

func TestDetectInMemoryReaders(t *testing.T) {
	pr, _ := io.Pipe()
	tests := []struct {
		r    io.Reader
		want bool
	}{
		{pr, false},

		{bytes.NewReader(nil), true},
		{bytes.NewBuffer(nil), true},
		{strings.NewReader(""), true},

		{io.NopCloser(pr), false},

		{io.NopCloser(bytes.NewReader(nil)), true},
		{io.NopCloser(bytes.NewBuffer(nil)), true},
		{io.NopCloser(strings.NewReader("")), true},
	}
	for i, tt := range tests {
		got := isKnownInMemoryReader(tt.r)
		if got != tt.want {
			t.Errorf("%d: got = %v; want %v", i, got, tt.want)
		}
	}
}

type mockTransferWriter struct {
	CalledReader io.Reader
	WriteCalled  bool
}

var _ io.ReaderFrom = (*mockTransferWriter)(nil)

func (w *mockTransferWriter) ReadFrom(r io.Reader) (int64, error) {
	w.CalledReader = r
	return io.Copy(io.Discard, r)
}

func (w *mockTransferWriter) Write(p []byte) (int, error) {
	w.WriteCalled = true
	return io.Discard.Write(p)
}

func TestTransferWriterWriteBodyReaderTypes(t *testing.T) {
	fileType := reflect.TypeFor[*os.File]()
	bufferType := reflect.TypeFor[*bytes.Buffer]()

	nBytes := int64(1 << 10)
	newFileFunc := func() (r io.Reader, done func(), err error) {
		f, err := os.CreateTemp("", "net-http-newfilefunc")
		if err != nil {
			return nil, nil, err
		}

		// Write some bytes to the file to enable reading.
		if _, err := io.CopyN(f, rand.Reader, nBytes); err != nil {
			return nil, nil, fmt.Errorf("failed to write data to file: %v", err)
		}
		if _, err := f.Seek(0, 0); err != nil {
			return nil, nil, fmt.Errorf("failed to seek to front: %v", err)
		}

		done = func() {
			f.Close()
			os.Remove(f.Name())
		}

		return f, done, nil
	}

	newBufferFunc := func() (io.Reader, func(), error) {
		return bytes.NewBuffer(make([]byte, nBytes)), func() {}, nil
	}

	cases := []struct {
		name             string
		bodyFunc         func() (io.Reader, func(), error)
		method           string
		contentLength    int64
		transferEncoding []string
		limitedReader    bool
		expectedReader   reflect.Type
		expectedWrite    bool
	}{
		{
			name:           "file, non-chunked, size set",
			bodyFunc:       newFileFunc,
			method:         "PUT",
			contentLength:  nBytes,
			limitedReader:  true,
			expectedReader: fileType,
		},
		{
			name:   "file, non-chunked, size set, nopCloser wrapped",
			method: "PUT",
			bodyFunc: func() (io.Reader, func(), error) {
				r, cleanup, err := newFileFunc()
				return io.NopCloser(r), cleanup, err
			},
			contentLength:  nBytes,
			limitedReader:  true,
			expectedReader: fileType,
		},
		{
			name:           "file, non-chunked, negative size",
			method:         "PUT",
			bodyFunc:       newFileFunc,
			contentLength:  -1,
			expectedReader: fileType,
		},
		{
			name:           "file, non-chunked, CONNECT, negative size",
			method:         "CONNECT",
			bodyFunc:       newFileFunc,
			contentLength:  -1,
			expectedReader: fileType,
		},
		{
			name:             "file, chunked",
			method:           "PUT",
			bodyFunc:         newFileFunc,
			transferEncoding: []string{"chunked"},
			expectedWrite:    true,
		},
		{
			name:           "buffer, non-chunked, size set",
			bodyFunc:       newBufferFunc,
			method:         "PUT",
			contentLength:  nBytes,
			limitedReader:  true,
			expectedReader: bufferType,
		},
		{
			name:   "buffer, non-chunked, size set, nopCloser wrapped",
			method: "PUT",
			bodyFunc: func() (io.Reader, func(), error) {
				r, cleanup, err := newBufferFunc()
				return io.NopCloser(r), cleanup, err
			},
			contentLength:  nBytes,
			limitedReader:  true,
			expectedReader: bufferType,
		},
		{
			name:          "buffer, non-chunked, negative size",
			method:        "PUT",
			bodyFunc:      newBufferFunc,
			contentLength: -1,
			expectedWrite: true,
		},
		{
			name:          "buffer, non-chunked, CONNECT, negative size",
			method:        "CONNECT",
			bodyFunc:      newBufferFunc,
			contentLength: -1,
			expectedWrite: true,
		},
		{
			name:             "buffer, chunked",
			method:           "PUT",
			bodyFunc:         newBufferFunc,
			transferEncoding: []string{"chunked"},
			expectedWrite:    true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body, cleanup, err := tc.bodyFunc()
			if err != nil {
				t.Fatal(err)
			}
			defer cleanup()

			mw := &mockTransferWriter{}
			tw := &transferWriter{
				Body:             body,
				ContentLength:    tc.contentLength,
				TransferEncoding: tc.transferEncoding,
			}

			if err := tw.writeBody(mw); err != nil {
				t.Fatal(err)
			}

			if tc.expectedReader != nil {
				if mw.CalledReader == nil {
					t.Fatal("did not call ReadFrom")
				}

				var actualReader reflect.Type
				lr, ok := mw.CalledReader.(*io.LimitedReader)
				if ok && tc.limitedReader {
					actualReader = reflect.TypeOf(lr.R)
				} else {
					actualReader = reflect.TypeOf(mw.CalledReader)
					// We have to handle this special case for genericWriteTo in os,
					// this struct is introduced to support a zero-copy optimization,
					// check out https://go.dev/issue/58808 for details.
					if actualReader.Kind() == reflect.Struct && actualReader.PkgPath() == "os" && actualReader.Name() == "fileWithoutWriteTo" {
						actualReader = actualReader.Field(1).Type
					}
				}

				if tc.expectedReader != actualReader {
					t.Fatalf("got reader %s want %s", actualReader, tc.expectedReader)
				}
			}

			if tc.expectedWrite && !mw.WriteCalled {
				t.Fatal("did not invoke Write")
			}
		})
	}
}

func TestParseTransferEncoding(t *testing.T) {
	tests := []struct {
		hdr     Header
		wantErr error
	}{
		{
			hdr:     Header{"Transfer-Encoding": {"fugazi"}},
			wantErr: &unsupportedTEError{`unsupported transfer encoding: "fugazi"`},
		},
		{
			hdr:     Header{"Transfer-Encoding": {"chunked, chunked", "identity", "chunked"}},
			wantErr: &unsupportedTEError{`too many transfer encodings: ["chunked, chunked" "identity" "chunked"]`},
		},
		{
			hdr:     Header{"Transfer-Encoding": {""}},
			wantErr: &unsupportedTEError{`unsupported transfer encoding: ""`},
		},
		{
			hdr:     Header{"Transfer-Encoding": {"chunked, identity"}},
			wantErr: &unsupportedTEError{`unsupported transfer encoding: "chunked, identity"`},
		},
		{
			hdr:     Header{"Transfer-Encoding": {"chunked", "identity"}},
			wantErr: &unsupportedTEError{`too many transfer encodings: ["chunked" "identity"]`},
		},
		{
			hdr:     Header{"Transfer-Encoding": {"\x0bchunked"}},
			wantErr: &unsupportedTEError{`unsupported transfer encoding: "\vchunked"`},
		},
		{
			hdr:     Header{"Transfer-Encoding": {"chunked"}},
			wantErr: nil,
		},
	}

	for i, tt := range tests {
		tr := &transferReader{
			Header:     tt.hdr,
			ProtoMajor: 1,
			ProtoMinor: 1,
		}
		gotErr := tr.parseTransferEncoding()
		if !reflect.DeepEqual(gotErr, tt.wantErr) {
			t.Errorf("%d.\ngot error:\n%v\nwant error:\n%v\n\n", i, gotErr, tt.wantErr)
		}
	}
}

// issue 39017 - disallow Content-Length values such as "+3"
func TestParseContentLength(t *testing.T) {
	tests := []struct {
		cl      string
		wantErr error
	}{
		{
			cl:      "",
			wantErr: badStringError("invalid empty Content-Length", ""),
		},
		{
			cl:      "3",
			wantErr: nil,
		},
		{
			cl:      "+3",
			wantErr: badStringError("bad Content-Length", "+3"),
		},
		{
			cl:      "-3",
			wantErr: badStringError("bad Content-Length", "-3"),
		},
		{
			// max int64, for safe conversion before returning
			cl:      "9223372036854775807",
			wantErr: nil,
		},
		{
			cl:      "9223372036854775808",
			wantErr: badStringError("bad Content-Length", "9223372036854775808"),
		},
	}

	for _, tt := range tests {
		if _, gotErr := parseContentLength([]string{tt.cl}); !reflect.DeepEqual(gotErr, tt.wantErr) {
			t.Errorf("%q:\n\tgot=%v\n\twant=%v", tt.cl, gotErr, tt.wantErr)
		}
	}
}
