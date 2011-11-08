// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"mime/multipart"
	. "net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"reflect"
	"regexp"
	"strings"
	"testing"
)

func TestQuery(t *testing.T) {
	req := &Request{Method: "GET"}
	req.URL, _ = url.Parse("http://www.google.com/search?q=foo&q=bar")
	if q := req.FormValue("q"); q != "foo" {
		t.Errorf(`req.FormValue("q") = %q, want "foo"`, q)
	}
}

func TestPostQuery(t *testing.T) {
	req, _ := NewRequest("POST", "http://www.google.com/search?q=foo&q=bar&both=x",
		strings.NewReader("z=post&both=y"))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded; param=value")

	if q := req.FormValue("q"); q != "foo" {
		t.Errorf(`req.FormValue("q") = %q, want "foo"`, q)
	}
	if z := req.FormValue("z"); z != "post" {
		t.Errorf(`req.FormValue("z") = %q, want "post"`, z)
	}
	if both := req.Form["both"]; !reflect.DeepEqual(both, []string{"x", "y"}) {
		t.Errorf(`req.FormValue("both") = %q, want ["x", "y"]`, both)
	}
}

type stringMap map[string][]string
type parseContentTypeTest struct {
	contentType stringMap
}

var parseContentTypeTests = []parseContentTypeTest{
	{contentType: stringMap{"Content-Type": {"text/plain"}}},
	{contentType: stringMap{}}, // Non-existent keys are not placed. The value nil is illegal.
	{contentType: stringMap{"Content-Type": {"text/plain; boundary="}}},
	{
		contentType: stringMap{"Content-Type": {"application/unknown"}},
	},
}

func TestParseFormBadContentType(t *testing.T) {
	for i, test := range parseContentTypeTests {
		req := &Request{
			Method: "POST",
			Header: Header(test.contentType),
			Body:   ioutil.NopCloser(bytes.NewBufferString("body")),
		}
		err := req.ParseForm()
		if err == nil {
			t.Errorf("test %d should have returned error", i)
		}
	}
}

func TestMultipartReader(t *testing.T) {
	req := &Request{
		Method: "POST",
		Header: Header{"Content-Type": {`multipart/form-data; boundary="foo123"`}},
		Body:   ioutil.NopCloser(new(bytes.Buffer)),
	}
	multipart, err := req.MultipartReader()
	if multipart == nil {
		t.Errorf("expected multipart; error: %v", err)
	}

	req.Header = Header{"Content-Type": {"text/plain"}}
	multipart, err = req.MultipartReader()
	if multipart != nil {
		t.Errorf("unexpected multipart for text/plain")
	}
}

func TestRedirect(t *testing.T) {
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		switch r.URL.Path {
		case "/":
			w.Header().Set("Location", "/foo/")
			w.WriteHeader(StatusSeeOther)
		case "/foo/":
			fmt.Fprintf(w, "foo")
		default:
			w.WriteHeader(StatusBadRequest)
		}
	}))
	defer ts.Close()

	var end = regexp.MustCompile("/foo/$")
	r, err := Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	url := r.Request.URL.String()
	if r.StatusCode != 200 || !end.MatchString(url) {
		t.Fatalf("Get got status %d at %q, want 200 matching /foo/$", r.StatusCode, url)
	}
}

func TestSetBasicAuth(t *testing.T) {
	r, _ := NewRequest("GET", "http://example.com/", nil)
	r.SetBasicAuth("Aladdin", "open sesame")
	if g, e := r.Header.Get("Authorization"), "Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ=="; g != e {
		t.Errorf("got header %q, want %q", g, e)
	}
}

func TestMultipartRequest(t *testing.T) {
	// Test that we can read the values and files of a 
	// multipart request with FormValue and FormFile,
	// and that ParseMultipartForm can be called multiple times.
	req := newTestMultipartRequest(t)
	if err := req.ParseMultipartForm(25); err != nil {
		t.Fatal("ParseMultipartForm first call:", err)
	}
	defer req.MultipartForm.RemoveAll()
	validateTestMultipartContents(t, req, false)
	if err := req.ParseMultipartForm(25); err != nil {
		t.Fatal("ParseMultipartForm second call:", err)
	}
	validateTestMultipartContents(t, req, false)
}

func TestMultipartRequestAuto(t *testing.T) {
	// Test that FormValue and FormFile automatically invoke
	// ParseMultipartForm and return the right values.
	req := newTestMultipartRequest(t)
	defer func() {
		if req.MultipartForm != nil {
			req.MultipartForm.RemoveAll()
		}
	}()
	validateTestMultipartContents(t, req, true)
}

func TestEmptyMultipartRequest(t *testing.T) {
	// Test that FormValue and FormFile automatically invoke
	// ParseMultipartForm and return the right values.
	req, err := NewRequest("GET", "/", nil)
	if err != nil {
		t.Errorf("NewRequest err = %q", err)
	}
	testMissingFile(t, req)
}

func TestRequestMultipartCallOrder(t *testing.T) {
	req := newTestMultipartRequest(t)
	_, err := req.MultipartReader()
	if err != nil {
		t.Fatalf("MultipartReader: %v", err)
	}
	err = req.ParseMultipartForm(1024)
	if err == nil {
		t.Errorf("expected an error from ParseMultipartForm after call to MultipartReader")
	}
}

func testMissingFile(t *testing.T, req *Request) {
	f, fh, err := req.FormFile("missing")
	if f != nil {
		t.Errorf("FormFile file = %q, want nil", f)
	}
	if fh != nil {
		t.Errorf("FormFile file header = %q, want nil", fh)
	}
	if err != ErrMissingFile {
		t.Errorf("FormFile err = %q, want ErrMissingFile", err)
	}
}

func newTestMultipartRequest(t *testing.T) *Request {
	b := bytes.NewBufferString(strings.Replace(message, "\n", "\r\n", -1))
	req, err := NewRequest("POST", "/", b)
	if err != nil {
		t.Fatal("NewRequest:", err)
	}
	ctype := fmt.Sprintf(`multipart/form-data; boundary="%s"`, boundary)
	req.Header.Set("Content-type", ctype)
	return req
}

func validateTestMultipartContents(t *testing.T, req *Request, allMem bool) {
	if g, e := req.FormValue("texta"), textaValue; g != e {
		t.Errorf("texta value = %q, want %q", g, e)
	}
	if g, e := req.FormValue("texta"), textaValue; g != e {
		t.Errorf("texta value = %q, want %q", g, e)
	}
	if g := req.FormValue("missing"); g != "" {
		t.Errorf("missing value = %q, want empty string", g)
	}

	assertMem := func(n string, fd multipart.File) {
		if _, ok := fd.(*os.File); ok {
			t.Error(n, " is *os.File, should not be")
		}
	}
	fd := testMultipartFile(t, req, "filea", "filea.txt", fileaContents)
	assertMem("filea", fd)
	fd = testMultipartFile(t, req, "fileb", "fileb.txt", filebContents)
	if allMem {
		assertMem("fileb", fd)
	} else {
		if _, ok := fd.(*os.File); !ok {
			t.Errorf("fileb has unexpected underlying type %T", fd)
		}
	}

	testMissingFile(t, req)
}

func testMultipartFile(t *testing.T, req *Request, key, expectFilename, expectContent string) multipart.File {
	f, fh, err := req.FormFile(key)
	if err != nil {
		t.Fatalf("FormFile(%q): %q", key, err)
	}
	if fh.Filename != expectFilename {
		t.Errorf("filename = %q, want %q", fh.Filename, expectFilename)
	}
	var b bytes.Buffer
	_, err = io.Copy(&b, f)
	if err != nil {
		t.Fatal("copying contents:", err)
	}
	if g := b.String(); g != expectContent {
		t.Errorf("contents = %q, want %q", g, expectContent)
	}
	return f
}

const (
	fileaContents = "This is a test file."
	filebContents = "Another test file."
	textaValue    = "foo"
	textbValue    = "bar"
	boundary      = `MyBoundary`
)

const message = `
--MyBoundary
Content-Disposition: form-data; name="filea"; filename="filea.txt"
Content-Type: text/plain

` + fileaContents + `
--MyBoundary
Content-Disposition: form-data; name="fileb"; filename="fileb.txt"
Content-Type: text/plain

` + filebContents + `
--MyBoundary
Content-Disposition: form-data; name="texta"

` + textaValue + `
--MyBoundary
Content-Disposition: form-data; name="textb"

` + textbValue + `
--MyBoundary--
`
