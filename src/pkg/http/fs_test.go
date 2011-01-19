// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"sync"
	"testing"
)

var ParseRangeTests = []struct {
	s      string
	length int64
	r      []httpRange
}{
	{"", 0, nil},
	{"foo", 0, nil},
	{"bytes=", 0, nil},
	{"bytes=5-4", 10, nil},
	{"bytes=0-2,5-4", 10, nil},
	{"bytes=0-9", 10, []httpRange{{0, 10}}},
	{"bytes=0-", 10, []httpRange{{0, 10}}},
	{"bytes=5-", 10, []httpRange{{5, 5}}},
	{"bytes=0-20", 10, []httpRange{{0, 10}}},
	{"bytes=15-,0-5", 10, nil},
	{"bytes=-5", 10, []httpRange{{5, 5}}},
	{"bytes=-15", 10, []httpRange{{0, 10}}},
	{"bytes=0-499", 10000, []httpRange{{0, 500}}},
	{"bytes=500-999", 10000, []httpRange{{500, 500}}},
	{"bytes=-500", 10000, []httpRange{{9500, 500}}},
	{"bytes=9500-", 10000, []httpRange{{9500, 500}}},
	{"bytes=0-0,-1", 10000, []httpRange{{0, 1}, {9999, 1}}},
	{"bytes=500-600,601-999", 10000, []httpRange{{500, 101}, {601, 399}}},
	{"bytes=500-700,601-999", 10000, []httpRange{{500, 201}, {601, 399}}},
}

func TestParseRange(t *testing.T) {
	for _, test := range ParseRangeTests {
		r := test.r
		ranges, err := parseRange(test.s, test.length)
		if err != nil && r != nil {
			t.Errorf("parseRange(%q) returned error %q", test.s, err)
		}
		if len(ranges) != len(r) {
			t.Errorf("len(parseRange(%q)) = %d, want %d", test.s, len(ranges), len(r))
			continue
		}
		for i := range r {
			if ranges[i].start != r[i].start {
				t.Errorf("parseRange(%q)[%d].start = %d, want %d", test.s, i, ranges[i].start, r[i].start)
			}
			if ranges[i].length != r[i].length {
				t.Errorf("parseRange(%q)[%d].length = %d, want %d", test.s, i, ranges[i].length, r[i].length)
			}
		}
	}
}

const (
	testFile       = "testdata/file"
	testFileLength = 11
)

var (
	serverOnce sync.Once
	serverAddr string
)

func startServer(t *testing.T) {
	serverOnce.Do(func() {
		HandleFunc("/ServeFile", func(w ResponseWriter, r *Request) {
			ServeFile(w, r, "testdata/file")
		})
		l, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal("listen:", err)
		}
		serverAddr = l.Addr().String()
		go Serve(l, nil)
	})
}

var ServeFileRangeTests = []struct {
	start, end int
	r          string
	code       int
}{
	{0, testFileLength, "", StatusOK},
	{0, 5, "0-4", StatusPartialContent},
	{2, testFileLength, "2-", StatusPartialContent},
	{testFileLength - 5, testFileLength, "-5", StatusPartialContent},
	{3, 8, "3-7", StatusPartialContent},
	{0, 0, "20-", StatusRequestedRangeNotSatisfiable},
}

func TestServeFile(t *testing.T) {
	startServer(t)
	var err os.Error

	file, err := ioutil.ReadFile(testFile)
	if err != nil {
		t.Fatal("reading file:", err)
	}

	// set up the Request (re-used for all tests)
	var req Request
	req.Header = make(map[string]string)
	if req.URL, err = ParseURL("http://" + serverAddr + "/ServeFile"); err != nil {
		t.Fatal("ParseURL:", err)
	}
	req.Method = "GET"

	// straight GET
	_, body := getBody(t, req)
	if !equal(body, file) {
		t.Fatalf("body mismatch: got %q, want %q", body, file)
	}

	// Range tests
	for _, rt := range ServeFileRangeTests {
		req.Header["Range"] = "bytes=" + rt.r
		if rt.r == "" {
			req.Header["Range"] = ""
		}
		r, body := getBody(t, req)
		if r.StatusCode != rt.code {
			t.Errorf("range=%q: StatusCode=%d, want %d", rt.r, r.StatusCode, rt.code)
		}
		if rt.code == StatusRequestedRangeNotSatisfiable {
			continue
		}
		h := fmt.Sprintf("bytes %d-%d/%d", rt.start, rt.end-1, testFileLength)
		if rt.r == "" {
			h = ""
		}
		if r.Header["Content-Range"] != h {
			t.Errorf("header mismatch: range=%q: got %q, want %q", rt.r, r.Header["Content-Range"], h)
		}
		if !equal(body, file[rt.start:rt.end]) {
			t.Errorf("body mismatch: range=%q: got %q, want %q", rt.r, body, file[rt.start:rt.end])
		}
	}
}

func getBody(t *testing.T, req Request) (*Response, []byte) {
	r, err := send(&req)
	if err != nil {
		t.Fatal(req.URL.String(), "send:", err)
	}
	b, err := ioutil.ReadAll(r.Body)
	if err != nil {
		t.Fatal("reading Body:", err)
	}
	return r, b
}

func equal(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
