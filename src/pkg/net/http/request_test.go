// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"bufio"
	"bytes"
	"encoding/base64"
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
	req, _ := NewRequest("POST", "http://www.google.com/search?q=foo&q=bar&both=x&prio=1&empty=not",
		strings.NewReader("z=post&both=y&prio=2&empty="))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded; param=value")

	if q := req.FormValue("q"); q != "foo" {
		t.Errorf(`req.FormValue("q") = %q, want "foo"`, q)
	}
	if z := req.FormValue("z"); z != "post" {
		t.Errorf(`req.FormValue("z") = %q, want "post"`, z)
	}
	if bq, found := req.PostForm["q"]; found {
		t.Errorf(`req.PostForm["q"] = %q, want no entry in map`, bq)
	}
	if bz := req.PostFormValue("z"); bz != "post" {
		t.Errorf(`req.PostFormValue("z") = %q, want "post"`, bz)
	}
	if qs := req.Form["q"]; !reflect.DeepEqual(qs, []string{"foo", "bar"}) {
		t.Errorf(`req.Form["q"] = %q, want ["foo", "bar"]`, qs)
	}
	if both := req.Form["both"]; !reflect.DeepEqual(both, []string{"y", "x"}) {
		t.Errorf(`req.Form["both"] = %q, want ["y", "x"]`, both)
	}
	if prio := req.FormValue("prio"); prio != "2" {
		t.Errorf(`req.FormValue("prio") = %q, want "2" (from body)`, prio)
	}
	if empty := req.FormValue("empty"); empty != "" {
		t.Errorf(`req.FormValue("empty") = %q, want "" (from body)`, empty)
	}
}

func TestPatchQuery(t *testing.T) {
	req, _ := NewRequest("PATCH", "http://www.google.com/search?q=foo&q=bar&both=x&prio=1&empty=not",
		strings.NewReader("z=post&both=y&prio=2&empty="))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded; param=value")

	if q := req.FormValue("q"); q != "foo" {
		t.Errorf(`req.FormValue("q") = %q, want "foo"`, q)
	}
	if z := req.FormValue("z"); z != "post" {
		t.Errorf(`req.FormValue("z") = %q, want "post"`, z)
	}
	if bq, found := req.PostForm["q"]; found {
		t.Errorf(`req.PostForm["q"] = %q, want no entry in map`, bq)
	}
	if bz := req.PostFormValue("z"); bz != "post" {
		t.Errorf(`req.PostFormValue("z") = %q, want "post"`, bz)
	}
	if qs := req.Form["q"]; !reflect.DeepEqual(qs, []string{"foo", "bar"}) {
		t.Errorf(`req.Form["q"] = %q, want ["foo", "bar"]`, qs)
	}
	if both := req.Form["both"]; !reflect.DeepEqual(both, []string{"y", "x"}) {
		t.Errorf(`req.Form["both"] = %q, want ["y", "x"]`, both)
	}
	if prio := req.FormValue("prio"); prio != "2" {
		t.Errorf(`req.FormValue("prio") = %q, want "2" (from body)`, prio)
	}
	if empty := req.FormValue("empty"); empty != "" {
		t.Errorf(`req.FormValue("empty") = %q, want "" (from body)`, empty)
	}
}

type stringMap map[string][]string
type parseContentTypeTest struct {
	shouldError bool
	contentType stringMap
}

var parseContentTypeTests = []parseContentTypeTest{
	{false, stringMap{"Content-Type": {"text/plain"}}},
	// Empty content type is legal - shoult be treated as
	// application/octet-stream (RFC 2616, section 7.2.1)
	{false, stringMap{}},
	{true, stringMap{"Content-Type": {"text/plain; boundary="}}},
	{false, stringMap{"Content-Type": {"application/unknown"}}},
}

func TestParseFormUnknownContentType(t *testing.T) {
	for i, test := range parseContentTypeTests {
		req := &Request{
			Method: "POST",
			Header: Header(test.contentType),
			Body:   ioutil.NopCloser(strings.NewReader("body")),
		}
		err := req.ParseForm()
		switch {
		case err == nil && test.shouldError:
			t.Errorf("test %d should have returned error", i)
		case err != nil && !test.shouldError:
			t.Errorf("test %d should not have returned error, got %v", i, err)
		}
	}
}

func TestParseFormInitializeOnError(t *testing.T) {
	nilBody, _ := NewRequest("POST", "http://www.google.com/search?q=foo", nil)
	tests := []*Request{
		nilBody,
		{Method: "GET", URL: nil},
	}
	for i, req := range tests {
		err := req.ParseForm()
		if req.Form == nil {
			t.Errorf("%d. Form not initialized, error %v", i, err)
		}
		if req.PostForm == nil {
			t.Errorf("%d. PostForm not initialized, error %v", i, err)
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
		t.Error("unexpected multipart for text/plain")
	}
}

func TestParseMultipartForm(t *testing.T) {
	req := &Request{
		Method: "POST",
		Header: Header{"Content-Type": {`multipart/form-data; boundary="foo123"`}},
		Body:   ioutil.NopCloser(new(bytes.Buffer)),
	}
	err := req.ParseMultipartForm(25)
	if err == nil {
		t.Error("expected multipart EOF, got nil")
	}

	req.Header = Header{"Content-Type": {"text/plain"}}
	err = req.ParseMultipartForm(25)
	if err != ErrNotMultipart {
		t.Error("expected ErrNotMultipart for text/plain")
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

func TestMissingFileMultipartRequest(t *testing.T) {
	// Test that FormFile returns an error if
	// the named file is missing.
	req := newTestMultipartRequest(t)
	testMissingFile(t, req)
}

// Test that FormValue invokes ParseMultipartForm.
func TestFormValueCallsParseMultipartForm(t *testing.T) {
	req, _ := NewRequest("POST", "http://www.google.com/", strings.NewReader("z=post"))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded; param=value")
	if req.Form != nil {
		t.Fatal("Unexpected request Form, want nil")
	}
	req.FormValue("z")
	if req.Form == nil {
		t.Fatal("ParseMultipartForm not called by FormValue")
	}
}

// Test that FormFile invokes ParseMultipartForm.
func TestFormFileCallsParseMultipartForm(t *testing.T) {
	req := newTestMultipartRequest(t)
	if req.Form != nil {
		t.Fatal("Unexpected request Form, want nil")
	}
	req.FormFile("")
	if req.Form == nil {
		t.Fatal("ParseMultipartForm not called by FormFile")
	}
}

// Test that ParseMultipartForm errors if called
// after MultipartReader on the same request.
func TestParseMultipartFormOrder(t *testing.T) {
	req := newTestMultipartRequest(t)
	if _, err := req.MultipartReader(); err != nil {
		t.Fatalf("MultipartReader: %v", err)
	}
	if err := req.ParseMultipartForm(1024); err == nil {
		t.Fatal("expected an error from ParseMultipartForm after call to MultipartReader")
	}
}

// Test that MultipartReader errors if called
// after ParseMultipartForm on the same request.
func TestMultipartReaderOrder(t *testing.T) {
	req := newTestMultipartRequest(t)
	if err := req.ParseMultipartForm(25); err != nil {
		t.Fatalf("ParseMultipartForm: %v", err)
	}
	defer req.MultipartForm.RemoveAll()
	if _, err := req.MultipartReader(); err == nil {
		t.Fatal("expected an error from MultipartReader after call to ParseMultipartForm")
	}
}

// Test that FormFile errors if called after
// MultipartReader on the same request.
func TestFormFileOrder(t *testing.T) {
	req := newTestMultipartRequest(t)
	if _, err := req.MultipartReader(); err != nil {
		t.Fatalf("MultipartReader: %v", err)
	}
	if _, _, err := req.FormFile(""); err == nil {
		t.Fatal("expected an error from FormFile after call to MultipartReader")
	}
}

var readRequestErrorTests = []struct {
	in  string
	err error
}{
	{"GET / HTTP/1.1\r\nheader:foo\r\n\r\n", nil},
	{"GET / HTTP/1.1\r\nheader:foo\r\n", io.ErrUnexpectedEOF},
	{"", io.EOF},
}

func TestReadRequestErrors(t *testing.T) {
	for i, tt := range readRequestErrorTests {
		_, err := ReadRequest(bufio.NewReader(strings.NewReader(tt.in)))
		if err != tt.err {
			t.Errorf("%d. got error = %v; want %v", i, err, tt.err)
		}
	}
}

func TestNewRequestHost(t *testing.T) {
	req, err := NewRequest("GET", "http://localhost:1234/", nil)
	if err != nil {
		t.Fatal(err)
	}
	if req.Host != "localhost:1234" {
		t.Errorf("Host = %q; want localhost:1234", req.Host)
	}
}

func TestNewRequestContentLength(t *testing.T) {
	readByte := func(r io.Reader) io.Reader {
		var b [1]byte
		r.Read(b[:])
		return r
	}
	tests := []struct {
		r    io.Reader
		want int64
	}{
		{bytes.NewReader([]byte("123")), 3},
		{bytes.NewBuffer([]byte("1234")), 4},
		{strings.NewReader("12345"), 5},
		// Not detected:
		{struct{ io.Reader }{strings.NewReader("xyz")}, 0},
		{io.NewSectionReader(strings.NewReader("x"), 0, 6), 0},
		{readByte(io.NewSectionReader(strings.NewReader("xy"), 0, 6)), 0},
	}
	for _, tt := range tests {
		req, err := NewRequest("POST", "http://localhost/", tt.r)
		if err != nil {
			t.Fatal(err)
		}
		if req.ContentLength != tt.want {
			t.Errorf("ContentLength(%T) = %d; want %d", tt.r, req.ContentLength, tt.want)
		}
	}
}

var parseHTTPVersionTests = []struct {
	vers         string
	major, minor int
	ok           bool
}{
	{"HTTP/0.9", 0, 9, true},
	{"HTTP/1.0", 1, 0, true},
	{"HTTP/1.1", 1, 1, true},
	{"HTTP/3.14", 3, 14, true},

	{"HTTP", 0, 0, false},
	{"HTTP/one.one", 0, 0, false},
	{"HTTP/1.1/", 0, 0, false},
	{"HTTP/-1,0", 0, 0, false},
	{"HTTP/0,-1", 0, 0, false},
	{"HTTP/", 0, 0, false},
	{"HTTP/1,1", 0, 0, false},
}

func TestParseHTTPVersion(t *testing.T) {
	for _, tt := range parseHTTPVersionTests {
		major, minor, ok := ParseHTTPVersion(tt.vers)
		if ok != tt.ok || major != tt.major || minor != tt.minor {
			type version struct {
				major, minor int
				ok           bool
			}
			t.Errorf("failed to parse %q, expected: %#v, got %#v", tt.vers, version{tt.major, tt.minor, tt.ok}, version{major, minor, ok})
		}
	}
}

type getBasicAuthTest struct {
	username, password string
	ok                 bool
}

type parseBasicAuthTest getBasicAuthTest

type basicAuthCredentialsTest struct {
	username, password string
}

var getBasicAuthTests = []struct {
	username, password string
	ok                 bool
}{
	{"Aladdin", "open sesame", true},
	{"Aladdin", "open:sesame", true},
	{"", "", true},
}

func TestGetBasicAuth(t *testing.T) {
	for _, tt := range getBasicAuthTests {
		r, _ := NewRequest("GET", "http://example.com/", nil)
		r.SetBasicAuth(tt.username, tt.password)
		username, password, ok := r.BasicAuth()
		if ok != tt.ok || username != tt.username || password != tt.password {
			t.Errorf("BasicAuth() = %#v, want %#v", getBasicAuthTest{username, password, ok},
				getBasicAuthTest{tt.username, tt.password, tt.ok})
		}
	}
	// Unauthenticated request.
	r, _ := NewRequest("GET", "http://example.com/", nil)
	username, password, ok := r.BasicAuth()
	if ok {
		t.Errorf("expected false from BasicAuth when the request is unauthenticated")
	}
	want := basicAuthCredentialsTest{"", ""}
	if username != want.username || password != want.password {
		t.Errorf("expected credentials: %#v when the request is unauthenticated, got %#v",
			want, basicAuthCredentialsTest{username, password})
	}
}

var parseBasicAuthTests = []struct {
	header, username, password string
	ok                         bool
}{
	{"Basic " + base64.StdEncoding.EncodeToString([]byte("Aladdin:open sesame")), "Aladdin", "open sesame", true},
	{"Basic " + base64.StdEncoding.EncodeToString([]byte("Aladdin:open:sesame")), "Aladdin", "open:sesame", true},
	{"Basic " + base64.StdEncoding.EncodeToString([]byte(":")), "", "", true},
	{"Basic" + base64.StdEncoding.EncodeToString([]byte("Aladdin:open sesame")), "", "", false},
	{base64.StdEncoding.EncodeToString([]byte("Aladdin:open sesame")), "", "", false},
	{"Basic ", "", "", false},
	{"Basic Aladdin:open sesame", "", "", false},
	{`Digest username="Aladdin"`, "", "", false},
}

func TestParseBasicAuth(t *testing.T) {
	for _, tt := range parseBasicAuthTests {
		r, _ := NewRequest("GET", "http://example.com/", nil)
		r.Header.Set("Authorization", tt.header)
		username, password, ok := r.BasicAuth()
		if ok != tt.ok || username != tt.username || password != tt.password {
			t.Errorf("BasicAuth() = %#v, want %#v", getBasicAuthTest{username, password, ok},
				getBasicAuthTest{tt.username, tt.password, tt.ok})
		}
	}
}

type logWrites struct {
	t   *testing.T
	dst *[]string
}

func (l logWrites) WriteByte(c byte) error {
	l.t.Fatalf("unexpected WriteByte call")
	return nil
}

func (l logWrites) Write(p []byte) (n int, err error) {
	*l.dst = append(*l.dst, string(p))
	return len(p), nil
}

func TestRequestWriteBufferedWriter(t *testing.T) {
	got := []string{}
	req, _ := NewRequest("GET", "http://foo.com/", nil)
	req.Write(logWrites{t, &got})
	want := []string{
		"GET / HTTP/1.1\r\n",
		"Host: foo.com\r\n",
		"User-Agent: " + DefaultUserAgent + "\r\n",
		"\r\n",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Writes = %q\n  Want = %q", got, want)
	}
}

func testMissingFile(t *testing.T, req *Request) {
	f, fh, err := req.FormFile("missing")
	if f != nil {
		t.Errorf("FormFile file = %v, want nil", f)
	}
	if fh != nil {
		t.Errorf("FormFile file header = %q, want nil", fh)
	}
	if err != ErrMissingFile {
		t.Errorf("FormFile err = %q, want ErrMissingFile", err)
	}
}

func newTestMultipartRequest(t *testing.T) *Request {
	b := strings.NewReader(strings.Replace(message, "\n", "\r\n", -1))
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
	if g, e := req.FormValue("textb"), textbValue; g != e {
		t.Errorf("textb value = %q, want %q", g, e)
	}
	if g := req.FormValue("missing"); g != "" {
		t.Errorf("missing value = %q, want empty string", g)
	}

	assertMem := func(n string, fd multipart.File) {
		if _, ok := fd.(*os.File); ok {
			t.Error(n, " is *os.File, should not be")
		}
	}
	fda := testMultipartFile(t, req, "filea", "filea.txt", fileaContents)
	defer fda.Close()
	assertMem("filea", fda)
	fdb := testMultipartFile(t, req, "fileb", "fileb.txt", filebContents)
	defer fdb.Close()
	if allMem {
		assertMem("fileb", fdb)
	} else {
		if _, ok := fdb.(*os.File); !ok {
			t.Errorf("fileb has unexpected underlying type %T", fdb)
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

func benchmarkReadRequest(b *testing.B, request string) {
	request = request + "\n"                             // final \n
	request = strings.Replace(request, "\n", "\r\n", -1) // expand \n to \r\n
	b.SetBytes(int64(len(request)))
	r := bufio.NewReader(&infiniteReader{buf: []byte(request)})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ReadRequest(r)
		if err != nil {
			b.Fatalf("failed to read request: %v", err)
		}
	}
}

// infiniteReader satisfies Read requests as if the contents of buf
// loop indefinitely.
type infiniteReader struct {
	buf    []byte
	offset int
}

func (r *infiniteReader) Read(b []byte) (int, error) {
	n := copy(b, r.buf[r.offset:])
	r.offset = (r.offset + n) % len(r.buf)
	return n, nil
}

func BenchmarkReadRequestChrome(b *testing.B) {
	// https://github.com/felixge/node-http-perf/blob/master/fixtures/get.http
	benchmarkReadRequest(b, `GET / HTTP/1.1
Host: localhost:8080
Connection: keep-alive
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_2) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.52 Safari/537.17
Accept-Encoding: gzip,deflate,sdch
Accept-Language: en-US,en;q=0.8
Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.3
Cookie: __utma=1.1978842379.1323102373.1323102373.1323102373.1; EPi:NumberOfVisits=1,2012-02-28T13:42:18; CrmSession=5b707226b9563e1bc69084d07a107c98; plushContainerWidth=100%25; plushNoTopMenu=0; hudson_auto_refresh=false
`)
}

func BenchmarkReadRequestCurl(b *testing.B) {
	// curl http://localhost:8080/
	benchmarkReadRequest(b, `GET / HTTP/1.1
User-Agent: curl/7.27.0
Host: localhost:8080
Accept: */*
`)
}

func BenchmarkReadRequestApachebench(b *testing.B) {
	// ab -n 1 -c 1 http://localhost:8080/
	benchmarkReadRequest(b, `GET / HTTP/1.0
Host: localhost:8080
User-Agent: ApacheBench/2.3
Accept: */*
`)
}

func BenchmarkReadRequestSiege(b *testing.B) {
	// siege -r 1 -c 1 http://localhost:8080/
	benchmarkReadRequest(b, `GET / HTTP/1.1
Host: localhost:8080
Accept: */*
Accept-Encoding: gzip
User-Agent: JoeDog/1.00 [en] (X11; I; Siege 2.70)
Connection: keep-alive
`)
}

func BenchmarkReadRequestWrk(b *testing.B) {
	// wrk -t 1 -r 1 -c 1 http://localhost:8080/
	benchmarkReadRequest(b, `GET / HTTP/1.1
Host: localhost:8080
`)
}
