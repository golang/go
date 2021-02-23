// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fcgi

import (
	"bytes"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
)

var sizeTests = []struct {
	size  uint32
	bytes []byte
}{
	{0, []byte{0x00}},
	{127, []byte{0x7F}},
	{128, []byte{0x80, 0x00, 0x00, 0x80}},
	{1000, []byte{0x80, 0x00, 0x03, 0xE8}},
	{33554431, []byte{0x81, 0xFF, 0xFF, 0xFF}},
}

func TestSize(t *testing.T) {
	b := make([]byte, 4)
	for i, test := range sizeTests {
		n := encodeSize(b, test.size)
		if !bytes.Equal(b[:n], test.bytes) {
			t.Errorf("%d expected %x, encoded %x", i, test.bytes, b)
		}
		size, n := readSize(test.bytes)
		if size != test.size {
			t.Errorf("%d expected %d, read %d", i, test.size, size)
		}
		if len(test.bytes) != n {
			t.Errorf("%d did not consume all the bytes", i)
		}
	}
}

var streamTests = []struct {
	desc    string
	recType recType
	reqId   uint16
	content []byte
	raw     []byte
}{
	{"single record", typeStdout, 1, nil,
		[]byte{1, byte(typeStdout), 0, 1, 0, 0, 0, 0},
	},
	// this data will have to be split into two records
	{"two records", typeStdin, 300, make([]byte, 66000),
		bytes.Join([][]byte{
			// header for the first record
			{1, byte(typeStdin), 0x01, 0x2C, 0xFF, 0xFF, 1, 0},
			make([]byte, 65536),
			// header for the second
			{1, byte(typeStdin), 0x01, 0x2C, 0x01, 0xD1, 7, 0},
			make([]byte, 472),
			// header for the empty record
			{1, byte(typeStdin), 0x01, 0x2C, 0, 0, 0, 0},
		},
			nil),
	},
}

type nilCloser struct {
	io.ReadWriter
}

func (c *nilCloser) Close() error { return nil }

func TestStreams(t *testing.T) {
	var rec record
outer:
	for _, test := range streamTests {
		buf := bytes.NewBuffer(test.raw)
		var content []byte
		for buf.Len() > 0 {
			if err := rec.read(buf); err != nil {
				t.Errorf("%s: error reading record: %v", test.desc, err)
				continue outer
			}
			content = append(content, rec.content()...)
		}
		if rec.h.Type != test.recType {
			t.Errorf("%s: got type %d expected %d", test.desc, rec.h.Type, test.recType)
			continue
		}
		if rec.h.Id != test.reqId {
			t.Errorf("%s: got request ID %d expected %d", test.desc, rec.h.Id, test.reqId)
			continue
		}
		if !bytes.Equal(content, test.content) {
			t.Errorf("%s: read wrong content", test.desc)
			continue
		}
		buf.Reset()
		c := newConn(&nilCloser{buf})
		w := newWriter(c, test.recType, test.reqId)
		if _, err := w.Write(test.content); err != nil {
			t.Errorf("%s: error writing record: %v", test.desc, err)
			continue
		}
		if err := w.Close(); err != nil {
			t.Errorf("%s: error closing stream: %v", test.desc, err)
			continue
		}
		if !bytes.Equal(buf.Bytes(), test.raw) {
			t.Errorf("%s: wrote wrong content", test.desc)
		}
	}
}

type writeOnlyConn struct {
	buf []byte
}

func (c *writeOnlyConn) Write(p []byte) (int, error) {
	c.buf = append(c.buf, p...)
	return len(p), nil
}

func (c *writeOnlyConn) Read(p []byte) (int, error) {
	return 0, errors.New("conn is write-only")
}

func (c *writeOnlyConn) Close() error {
	return nil
}

func TestGetValues(t *testing.T) {
	var rec record
	rec.h.Type = typeGetValues

	wc := new(writeOnlyConn)
	c := newChild(wc, nil)
	err := c.handleRecord(&rec)
	if err != nil {
		t.Fatalf("handleRecord: %v", err)
	}

	const want = "\x01\n\x00\x00\x00\x12\x06\x00" +
		"\x0f\x01FCGI_MPXS_CONNS1" +
		"\x00\x00\x00\x00\x00\x00\x01\n\x00\x00\x00\x00\x00\x00"
	if got := string(wc.buf); got != want {
		t.Errorf(" got: %q\nwant: %q\n", got, want)
	}
}

func nameValuePair11(nameData, valueData string) []byte {
	return bytes.Join(
		[][]byte{
			{byte(len(nameData)), byte(len(valueData))},
			[]byte(nameData),
			[]byte(valueData),
		},
		nil,
	)
}

func makeRecord(
	recordType recType,
	requestId uint16,
	contentData []byte,
) []byte {
	requestIdB1 := byte(requestId >> 8)
	requestIdB0 := byte(requestId)

	contentLength := len(contentData)
	contentLengthB1 := byte(contentLength >> 8)
	contentLengthB0 := byte(contentLength)
	return bytes.Join([][]byte{
		{1, byte(recordType), requestIdB1, requestIdB0, contentLengthB1,
			contentLengthB0, 0, 0},
		contentData,
	},
		nil)
}

// a series of FastCGI records that start a request and begin sending the
// request body
var streamBeginTypeStdin = bytes.Join([][]byte{
	// set up request 1
	makeRecord(typeBeginRequest, 1,
		[]byte{0, byte(roleResponder), 0, 0, 0, 0, 0, 0}),
	// add required parameters to request 1
	makeRecord(typeParams, 1, nameValuePair11("REQUEST_METHOD", "GET")),
	makeRecord(typeParams, 1, nameValuePair11("SERVER_PROTOCOL", "HTTP/1.1")),
	makeRecord(typeParams, 1, nil),
	// begin sending body of request 1
	makeRecord(typeStdin, 1, []byte("0123456789abcdef")),
},
	nil)

var cleanUpTests = []struct {
	input []byte
	err   error
}{
	// confirm that child.handleRecord closes req.pw after aborting req
	{
		bytes.Join([][]byte{
			streamBeginTypeStdin,
			makeRecord(typeAbortRequest, 1, nil),
		},
			nil),
		ErrRequestAborted,
	},
	// confirm that child.serve closes all pipes after error reading record
	{
		bytes.Join([][]byte{
			streamBeginTypeStdin,
			nil,
		},
			nil),
		ErrConnClosed,
	},
}

type nopWriteCloser struct {
	io.Reader
}

func (nopWriteCloser) Write(buf []byte) (int, error) {
	return len(buf), nil
}

func (nopWriteCloser) Close() error {
	return nil
}

// Test that child.serve closes the bodies of aborted requests and closes the
// bodies of all requests before returning. Causes deadlock if either condition
// isn't met. See issue 6934.
func TestChildServeCleansUp(t *testing.T) {
	for _, tt := range cleanUpTests {
		input := make([]byte, len(tt.input))
		copy(input, tt.input)
		rc := nopWriteCloser{bytes.NewReader(input)}
		done := make(chan bool)
		c := newChild(rc, http.HandlerFunc(func(
			w http.ResponseWriter,
			r *http.Request,
		) {
			// block on reading body of request
			_, err := io.Copy(io.Discard, r.Body)
			if err != tt.err {
				t.Errorf("Expected %#v, got %#v", tt.err, err)
			}
			// not reached if body of request isn't closed
			done <- true
		}))
		go c.serve()
		// wait for body of request to be closed or all goroutines to block
		<-done
	}
}

type rwNopCloser struct {
	io.Reader
	io.Writer
}

func (rwNopCloser) Close() error {
	return nil
}

// Verifies it doesn't crash. 	Issue 11824.
func TestMalformedParams(t *testing.T) {
	input := []byte{
		// beginRequest, requestId=1, contentLength=8, role=1, keepConn=1
		1, 1, 0, 1, 0, 8, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
		// params, requestId=1, contentLength=10, k1Len=50, v1Len=50 (malformed, wrong length)
		1, 4, 0, 1, 0, 10, 0, 0, 50, 50, 3, 4, 5, 6, 7, 8, 9, 10,
		// end of params
		1, 4, 0, 1, 0, 0, 0, 0,
	}
	rw := rwNopCloser{bytes.NewReader(input), io.Discard}
	c := newChild(rw, http.DefaultServeMux)
	c.serve()
}

// a series of FastCGI records that start and end a request
var streamFullRequestStdin = bytes.Join([][]byte{
	// set up request
	makeRecord(typeBeginRequest, 1,
		[]byte{0, byte(roleResponder), 0, 0, 0, 0, 0, 0}),
	// add required parameters
	makeRecord(typeParams, 1, nameValuePair11("REQUEST_METHOD", "GET")),
	makeRecord(typeParams, 1, nameValuePair11("SERVER_PROTOCOL", "HTTP/1.1")),
	// set optional parameters
	makeRecord(typeParams, 1, nameValuePair11("REMOTE_USER", "jane.doe")),
	makeRecord(typeParams, 1, nameValuePair11("QUERY_STRING", "/foo/bar")),
	makeRecord(typeParams, 1, nil),
	// begin sending body of request
	makeRecord(typeStdin, 1, []byte("0123456789abcdef")),
	// end request
	makeRecord(typeEndRequest, 1, nil),
},
	nil)

var envVarTests = []struct {
	input               []byte
	envVar              string
	expectedVal         string
	expectedFilteredOut bool
}{
	{
		streamFullRequestStdin,
		"REMOTE_USER",
		"jane.doe",
		false,
	},
	{
		streamFullRequestStdin,
		"QUERY_STRING",
		"",
		true,
	},
}

// Test that environment variables set for a request can be
// read by a handler. Ensures that variables not set will not
// be exposed to a handler.
func TestChildServeReadsEnvVars(t *testing.T) {
	for _, tt := range envVarTests {
		input := make([]byte, len(tt.input))
		copy(input, tt.input)
		rc := nopWriteCloser{bytes.NewReader(input)}
		done := make(chan bool)
		c := newChild(rc, http.HandlerFunc(func(
			w http.ResponseWriter,
			r *http.Request,
		) {
			env := ProcessEnv(r)
			if _, ok := env[tt.envVar]; ok && tt.expectedFilteredOut {
				t.Errorf("Expected environment variable %s to not be set, but set to %s",
					tt.envVar, env[tt.envVar])
			} else if env[tt.envVar] != tt.expectedVal {
				t.Errorf("Expected %s, got %s", tt.expectedVal, env[tt.envVar])
			}
			done <- true
		}))
		go c.serve()
		<-done
	}
}

func TestResponseWriterSniffsContentType(t *testing.T) {
	var tests = []struct {
		name   string
		body   string
		wantCT string
	}{
		{
			name:   "no body",
			wantCT: "text/plain; charset=utf-8",
		},
		{
			name:   "html",
			body:   "<html><head><title>test page</title></head><body>This is a body</body></html>",
			wantCT: "text/html; charset=utf-8",
		},
		{
			name:   "text",
			body:   strings.Repeat("gopher", 86),
			wantCT: "text/plain; charset=utf-8",
		},
		{
			name:   "jpg",
			body:   "\xFF\xD8\xFF" + strings.Repeat("B", 1024),
			wantCT: "image/jpeg",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := make([]byte, len(streamFullRequestStdin))
			copy(input, streamFullRequestStdin)
			rc := nopWriteCloser{bytes.NewReader(input)}
			done := make(chan bool)
			var resp *response
			c := newChild(rc, http.HandlerFunc(func(
				w http.ResponseWriter,
				r *http.Request,
			) {
				io.WriteString(w, tt.body)
				resp = w.(*response)
				done <- true
			}))
			defer c.cleanUp()
			go c.serve()
			<-done
			if got := resp.Header().Get("Content-Type"); got != tt.wantCT {
				t.Errorf("got a Content-Type of %q; expected it to start with %q", got, tt.wantCT)
			}
		})
	}
}
