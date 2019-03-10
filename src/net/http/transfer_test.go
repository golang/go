// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"crypto/rand"
	"fmt"
	"io"
	"io/ioutil"
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

		{ioutil.NopCloser(pr), false},

		{ioutil.NopCloser(bytes.NewReader(nil)), true},
		{ioutil.NopCloser(bytes.NewBuffer(nil)), true},
		{ioutil.NopCloser(strings.NewReader("")), true},
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
	return io.Copy(ioutil.Discard, r)
}

func (w *mockTransferWriter) Write(p []byte) (int, error) {
	w.WriteCalled = true
	return ioutil.Discard.Write(p)
}

func TestTransferWriterWriteBodyReaderTypes(t *testing.T) {
	fileType := reflect.TypeOf(&os.File{})
	bufferType := reflect.TypeOf(&bytes.Buffer{})

	nBytes := int64(1 << 10)
	newFileFunc := func() (r io.Reader, done func(), err error) {
		f, err := ioutil.TempFile("", "net-http-newfilefunc")
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
				return ioutil.NopCloser(r), cleanup, err
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
				return ioutil.NopCloser(r), cleanup, err
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
				}

				if tc.expectedReader != actualReader {
					t.Fatalf("got reader %T want %T", actualReader, tc.expectedReader)
				}
			}

			if tc.expectedWrite && !mw.WriteCalled {
				t.Fatal("did not invoke Write")
			}
		})
	}
}

func TestFixTransferEncoding(t *testing.T) {
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
			wantErr: &badStringError{"chunked must be applied only once, as the last encoding", "chunked, chunked"},
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
		gotErr := tr.fixTransferEncoding()
		if !reflect.DeepEqual(gotErr, tt.wantErr) {
			t.Errorf("%d.\ngot error:\n%v\nwant error:\n%v\n\n", i, gotErr, tt.wantErr)
		}
	}
}

func gzipIt(s string) string {
	buf := new(bytes.Buffer)
	gw := gzip.NewWriter(buf)
	gw.Write([]byte(s))
	gw.Close()
	return buf.String()
}

func TestUnitTestProxyingReadCloserClosesBody(t *testing.T) {
	var checker closeChecker
	buf := new(bytes.Buffer)
	buf.WriteString("Hello, Gophers!")
	prc := &proxyingReadCloser{
		Reader: buf,
		Closer: &checker,
	}
	prc.Close()

	read, err := ioutil.ReadAll(prc)
	if err != nil {
		t.Fatalf("Read error: %v", err)
	}
	if g, w := string(read), "Hello, Gophers!"; g != w {
		t.Errorf("Read mismatch: got %q want %q", g, w)
	}

	if checker.closed != true {
		t.Fatal("closeChecker.Close was never invoked")
	}
}

func TestGzipTransferEncoding_request(t *testing.T) {
	helloWorldGzipped := gzipIt("Hello, World!")

	tests := []struct {
		payload  string
		wantErr  string
		wantBody string
	}{

		{
			// The case of "chunked" properly applied as the last encoding
			// and a gzipped request payload that is streamed in 3 parts.
			payload: `POST / HTTP/1.1
Host: golang.org
Transfer-Encoding: gzip, chunked
Content-Type: text/html; charset=UTF-8

` + fmt.Sprintf("%02x\r\n%s\r\n%02x\r\n%s\r\n%02x\r\n%s\r\n0\r\n\r\n",
				3, helloWorldGzipped[:3],
				5, helloWorldGzipped[3:8],
				len(helloWorldGzipped)-8, helloWorldGzipped[8:]),
			wantBody: `Hello, World!`,
		},

		{
			// The request specifies "Transfer-Encoding: chunked" so its body must be left untouched.
			payload: `PUT / HTTP/1.1
Host: golang.org
Transfer-Encoding: chunked
Connection: close
Content-Type: text/html; charset=UTF-8

` + fmt.Sprintf("%0x\r\n%s\r\n0\r\n\r\n", len(helloWorldGzipped), helloWorldGzipped),
			// We want that payload as it was sent.
			wantBody: helloWorldGzipped,
		},

		{
			// Valid request, the body doesn't have "Transfer-Encoding: chunked" but implicitly encoded
			// for chunking as per the advisory from RFC 7230 3.3.1 which advises for cases where.
			payload: `POST / HTTP/1.1
Host: localhost
Transfer-Encoding: gzip
Content-Type: text/html; charset=UTF-8

` + fmt.Sprintf("%0x\r\n%s\r\n0\r\n\r\n", len(helloWorldGzipped), helloWorldGzipped),
			wantBody: `Hello, World!`,
		},

		{
			// Invalid request, the body isn't chunked nor is the connection terminated immediately
			// hence invalid as per the advisory from RFC 7230 3.3.1 which advises for cases where
			// a Transfer-Encoding that isn't finally chunked is provided.
			payload: `PUT / HTTP/1.1
Host: golang.org
Transfer-Encoding: gzip
Content-Length: 0
Connection: close
Content-Type: text/html; charset=UTF-8

`,
			wantErr: `EOF`,
		},

		{
			// The case of chunked applied before another encoding.
			payload: `PUT / HTTP/1.1
Location: golang.org
Transfer-Encoding: chunked, gzip
Content-Length: 0
Connection: close
Content-Type: text/html; charset=UTF-8

`,
			wantErr: `chunked must be applied only once, as the last encoding "chunked, gzip"`,
		},

		{
			// The case of chunked properly applied as the
			// last encoding BUT with a bad "Content-Length".
			payload: `POST / HTTP/1.1
Host: golang.org
Transfer-Encoding: gzip, chunked
Content-Length: 10
Connection: close
Content-Type: text/html; charset=UTF-8

` + "0\r\n\r\n",
			wantErr: "EOF",
		},
	}

	for i, tt := range tests {
		req, err := ReadRequest(bufio.NewReader(strings.NewReader(tt.payload)))
		if tt.wantErr != "" {
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("test %d. Error mismatch\nGot:  %v\nWant: %s", i, err, tt.wantErr)
			}
			continue
		}

		if err != nil {
			t.Errorf("test %d. Unexpected ReadRequest error: %v\nPayload:\n%s", i, err, tt.payload)
			continue
		}

		got, err := ioutil.ReadAll(req.Body)
		req.Body.Close()
		if err != nil {
			t.Errorf("test %d. Failed to read response body: %v", i, err)
		}
		if g, w := string(got), tt.wantBody; g != w {
			t.Errorf("test %d. Request body mimsatch\nGot:\n%s\n\nWant:\n%s", i, g, w)
		}
	}
}

func TestGzipTransferEncoding_response(t *testing.T) {
	helloWorldGzipped := gzipIt("Hello, World!")

	tests := []struct {
		payload  string
		wantErr  string
		wantBody string
	}{

		{
			// The case of "chunked" properly applied as the last encoding
			// and a gzipped payload that is streamed in 3 parts.
			payload: `HTTP/1.1 302 Found
Location: https://golang.org/
Transfer-Encoding: gzip, chunked
Connection: close
Content-Type: text/html; charset=UTF-8

` + fmt.Sprintf("%02x\r\n%s\r\n%02x\r\n%s\r\n%02x\r\n%s\r\n0\r\n\r\n",
				3, helloWorldGzipped[:3],
				5, helloWorldGzipped[3:8],
				len(helloWorldGzipped)-8, helloWorldGzipped[8:]),
			wantBody: `Hello, World!`,
		},

		{
			// The response specifies "Transfer-Encoding: chunked" so response body must be left untouched.
			payload: `HTTP/1.1 302 Found
Location: https://golang.org/
Transfer-Encoding: chunked
Connection: close
Content-Type: text/html; charset=UTF-8

` + fmt.Sprintf("%0x\r\n%s\r\n0\r\n\r\n", len(helloWorldGzipped), helloWorldGzipped),
			// We want that payload as it was sent.
			wantBody: helloWorldGzipped,
		},

		{
			// Valid response, the body doesn't have "Transfer-Encoding: chunked" but implicitly encoded
			// for chunking as per the advisory from RFC 7230 3.3.1 which advises for cases where.
			payload: `HTTP/1.1 302 Found
Location: https://golang.org/
Transfer-Encoding: gzip
Connection: close
Content-Type: text/html; charset=UTF-8

` + fmt.Sprintf("%0x\r\n%s\r\n0\r\n\r\n", len(helloWorldGzipped), helloWorldGzipped),
			wantBody: `Hello, World!`,
		},

		{
			// Invalid response, the body isn't chunked nor is the connection terminated immediately
			// hence invalid as per the advisory from RFC 7230 3.3.1 which advises for cases where
			// a Transfer-Encoding that isn't finally chunked is provided.
			payload: `HTTP/1.1 302 Found
Location: https://golang.org/
Transfer-Encoding: gzip
Content-Length: 0
Connection: close
Content-Type: text/html; charset=UTF-8

`,
			wantErr: `EOF`,
		},

		{
			// The case of chunked applied before another encoding.
			payload: `HTTP/1.1 302 Found
Location: https://golang.org/
Transfer-Encoding: chunked, gzip
Content-Length: 0
Connection: close
Content-Type: text/html; charset=UTF-8

`,
			wantErr: `chunked must be applied only once, as the last encoding "chunked, gzip"`,
		},

		{
			// The case of chunked properly applied as the
			// last encoding BUT with a bad "Content-Length".
			payload: `HTTP/1.1 302 Found
Location: https://golang.org/
Transfer-Encoding: gzip, chunked
Content-Length: 10
Connection: close
Content-Type: text/html; charset=UTF-8

` + "0\r\n\r\n",
			wantErr: "EOF",
		},

		{
			// Including "identity" more than once.
			payload: `HTTP/1.1 200 OK
Location: https://golang.org/
Transfer-Encoding: identity, identity
Content-Length: 0
Connection: close
Content-Type: text/html; charset=UTF-8

` + "0\r\n\r\n",
			wantErr: `"identity" when present must be the only transfer encoding "identity, identity"`,
		},
	}

	for i, tt := range tests {
		res, err := ReadResponse(bufio.NewReader(strings.NewReader(tt.payload)), nil)
		if tt.wantErr != "" {
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("test %d. Error mismatch\nGot:  %v\nWant: %s", i, err, tt.wantErr)
			}
			continue
		}

		if err != nil {
			t.Errorf("test %d. Unexpected ReadResponse error: %v\nPayload:\n%s", i, err, tt.payload)
			continue
		}

		got, err := ioutil.ReadAll(res.Body)
		res.Body.Close()
		if err != nil {
			t.Errorf("test %d. Failed to read response body: %v", i, err)
		}
		if g, w := string(got), tt.wantBody; g != w {
			t.Errorf("test %d. Response body mimsatch\nGot:\n%s\n\nWant:\n%s", i, g, w)
		}
	}
}
