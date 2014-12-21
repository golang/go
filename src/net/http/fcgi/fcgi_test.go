// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fcgi

import (
	"bytes"
	"errors"
	"io"
	"io/ioutil"
	"net/http"
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
	io.ReadWriter
}

func (nopWriteCloser) Close() error {
	return nil
}

// Test that child.serve closes the bodies of aborted requests and closes the
// bodies of all requests before returning. Causes deadlock if either condition
// isn't met. See issue 6934.
func TestChildServeCleansUp(t *testing.T) {
	for _, tt := range cleanUpTests {
		rc := nopWriteCloser{bytes.NewBuffer(tt.input)}
		done := make(chan bool)
		c := newChild(rc, http.HandlerFunc(func(
			w http.ResponseWriter,
			r *http.Request,
		) {
			// block on reading body of request
			_, err := io.Copy(ioutil.Discard, r.Body)
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
