// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"fmt"
	. "http"
	"http/httptest"
	"io/ioutil"
	"os"
	"testing"
)

const (
	testFile       = "testdata/file"
	testFileLength = 11
)

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
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		ServeFile(w, r, "testdata/file")
	}))
	defer ts.Close()

	var err os.Error

	file, err := ioutil.ReadFile(testFile)
	if err != nil {
		t.Fatal("reading file:", err)
	}

	// set up the Request (re-used for all tests)
	var req Request
	req.Header = make(Header)
	if req.URL, err = ParseURL(ts.URL); err != nil {
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
		req.Header.Set("Range", "bytes="+rt.r)
		if rt.r == "" {
			req.Header["Range"] = nil
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
		cr := r.Header.Get("Content-Range")
		if cr != h {
			t.Errorf("header mismatch: range=%q: got %q, want %q", rt.r, cr, h)
		}
		if !equal(body, file[rt.start:rt.end]) {
			t.Errorf("body mismatch: range=%q: got %q, want %q", rt.r, body, file[rt.start:rt.end])
		}
	}
}

func TestServeFileContentType(t *testing.T) {
	const ctype = "icecream/chocolate"
	override := false
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		if override {
			w.Header().Set("Content-Type", ctype)
		}
		ServeFile(w, r, "testdata/file")
	}))
	defer ts.Close()
	get := func(want string) {
		resp, _, err := Get(ts.URL)
		if err != nil {
			t.Fatal(err)
		}
		if h := resp.Header.Get("Content-Type"); h != want {
			t.Errorf("Content-Type mismatch: got %q, want %q", h, want)
		}
	}
	get("text/plain; charset=utf-8")
	override = true
	get(ctype)
}

func getBody(t *testing.T, req Request) (*Response, []byte) {
	r, err := DefaultClient.Do(&req)
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
