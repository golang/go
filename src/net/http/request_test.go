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

func TestParseFormQuery(t *testing.T) {
	req, _ := NewRequest("POST", "http://www.google.com/search?q=foo&q=bar&both=x&prio=1&orphan=nope&empty=not",
		strings.NewReader("z=post&both=y&prio=2&=nokey&orphan;empty=&"))
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
	if orphan := req.Form["orphan"]; !reflect.DeepEqual(orphan, []string{"", "nope"}) {
		t.Errorf(`req.FormValue("orphan") = %q, want "" (from body)`, orphan)
	}
	if empty := req.Form["empty"]; !reflect.DeepEqual(empty, []string{"", "not"}) {
		t.Errorf(`req.FormValue("empty") = %q, want "" (from body)`, empty)
	}
	if nokey := req.Form[""]; !reflect.DeepEqual(nokey, []string{"nokey"}) {
		t.Errorf(`req.FormValue("nokey") = %q, want "nokey" (from body)`, nokey)
	}
}

// Tests that we only parse the form automatically for certain methods.
func TestParseFormQueryMethods(t *testing.T) {
	for _, method := range []string{"POST", "PATCH", "PUT", "FOO"} {
		req, _ := NewRequest(method, "http://www.google.com/search",
			strings.NewReader("foo=bar"))
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded; param=value")
		want := "bar"
		if method == "FOO" {
			want = ""
		}
		if got := req.FormValue("foo"); got != want {
			t.Errorf(`for method %s, FormValue("foo") = %q; want %q`, method, got, want)
		}
	}
}

type stringMap map[string][]string
type parseContentTypeTest struct {
	shouldError bool
	contentType stringMap
}

var parseContentTypeTests = []parseContentTypeTest{
	{false, stringMap{"Content-Type": {"text/plain"}}},
	// Empty content type is legal - should be treated as
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

// Issue 9305: ParseMultipartForm should populate PostForm too
func TestParseMultipartFormPopulatesPostForm(t *testing.T) {
	postData :=
		`--xxx
Content-Disposition: form-data; name="field1"

value1
--xxx
Content-Disposition: form-data; name="field2"

value2
--xxx
Content-Disposition: form-data; name="file"; filename="file"
Content-Type: application/octet-stream
Content-Transfer-Encoding: binary

binary data
--xxx--
`
	req := &Request{
		Method: "POST",
		Header: Header{"Content-Type": {`multipart/form-data; boundary=xxx`}},
		Body:   ioutil.NopCloser(strings.NewReader(postData)),
	}

	initialFormItems := map[string]string{
		"language": "Go",
		"name":     "gopher",
		"skill":    "go-ing",
		"field2":   "initial-value2",
	}

	req.Form = make(url.Values)
	for k, v := range initialFormItems {
		req.Form.Add(k, v)
	}

	err := req.ParseMultipartForm(10000)
	if err != nil {
		t.Fatalf("unexpected multipart error %v", err)
	}

	wantForm := url.Values{
		"language": []string{"Go"},
		"name":     []string{"gopher"},
		"skill":    []string{"go-ing"},
		"field1":   []string{"value1"},
		"field2":   []string{"initial-value2", "value2"},
	}
	if !reflect.DeepEqual(req.Form, wantForm) {
		t.Fatalf("req.Form = %v, want %v", req.Form, wantForm)
	}

	wantPostForm := url.Values{
		"field1": []string{"value1"},
		"field2": []string{"value2"},
	}
	if !reflect.DeepEqual(req.PostForm, wantPostForm) {
		t.Fatalf("req.PostForm = %v, want %v", req.PostForm, wantPostForm)
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

func TestRedirect_h1(t *testing.T) { testRedirect(t, h1Mode) }
func TestRedirect_h2(t *testing.T) { testRedirect(t, h2Mode) }
func testRedirect(t *testing.T, h2 bool) {
	defer afterTest(t)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
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
	defer cst.close()

	var end = regexp.MustCompile("/foo/$")
	r, err := cst.c.Get(cst.ts.URL)
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
	err string

	header Header
}{
	0: {"GET / HTTP/1.1\r\nheader:foo\r\n\r\n", "", Header{"Header": {"foo"}}},
	1: {"GET / HTTP/1.1\r\nheader:foo\r\n", io.ErrUnexpectedEOF.Error(), nil},
	2: {"", io.EOF.Error(), nil},
	3: {
		in:  "HEAD / HTTP/1.1\r\nContent-Length:4\r\n\r\n",
		err: "http: method cannot contain a Content-Length",
	},
	4: {
		in:     "HEAD / HTTP/1.1\r\n\r\n",
		header: Header{},
	},

	// Multiple Content-Length values should either be
	// deduplicated if same or reject otherwise
	// See Issue 16490.
	5: {
		in:  "POST / HTTP/1.1\r\nContent-Length: 10\r\nContent-Length: 0\r\n\r\nGopher hey\r\n",
		err: "cannot contain multiple Content-Length headers",
	},
	6: {
		in:  "POST / HTTP/1.1\r\nContent-Length: 10\r\nContent-Length: 6\r\n\r\nGopher\r\n",
		err: "cannot contain multiple Content-Length headers",
	},
	7: {
		in:     "PUT / HTTP/1.1\r\nContent-Length: 6 \r\nContent-Length: 6\r\nContent-Length:6\r\n\r\nGopher\r\n",
		err:    "",
		header: Header{"Content-Length": {"6"}},
	},
	8: {
		in:  "PUT / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 6 \r\n\r\n",
		err: "cannot contain multiple Content-Length headers",
	},
	9: {
		in:  "POST / HTTP/1.1\r\nContent-Length:\r\nContent-Length: 3\r\n\r\n",
		err: "cannot contain multiple Content-Length headers",
	},
	10: {
		in:     "HEAD / HTTP/1.1\r\nContent-Length:0\r\nContent-Length: 0\r\n\r\n",
		header: Header{"Content-Length": {"0"}},
	},
}

func TestReadRequestErrors(t *testing.T) {
	for i, tt := range readRequestErrorTests {
		req, err := ReadRequest(bufio.NewReader(strings.NewReader(tt.in)))
		if err == nil {
			if tt.err != "" {
				t.Errorf("#%d: got nil err; want %q", i, tt.err)
			}

			if !reflect.DeepEqual(tt.header, req.Header) {
				t.Errorf("#%d: gotHeader: %q wantHeader: %q", i, req.Header, tt.header)
			}
			continue
		}

		if tt.err == "" || !strings.Contains(err.Error(), tt.err) {
			t.Errorf("%d: got error = %v; want %v", i, err, tt.err)
		}
	}
}

var newRequestHostTests = []struct {
	in, out string
}{
	{"http://www.example.com/", "www.example.com"},
	{"http://www.example.com:8080/", "www.example.com:8080"},

	{"http://192.168.0.1/", "192.168.0.1"},
	{"http://192.168.0.1:8080/", "192.168.0.1:8080"},
	{"http://192.168.0.1:/", "192.168.0.1"},

	{"http://[fe80::1]/", "[fe80::1]"},
	{"http://[fe80::1]:8080/", "[fe80::1]:8080"},
	{"http://[fe80::1%25en0]/", "[fe80::1%en0]"},
	{"http://[fe80::1%25en0]:8080/", "[fe80::1%en0]:8080"},
	{"http://[fe80::1%25en0]:/", "[fe80::1%en0]"},
}

func TestNewRequestHost(t *testing.T) {
	for i, tt := range newRequestHostTests {
		req, err := NewRequest("GET", tt.in, nil)
		if err != nil {
			t.Errorf("#%v: %v", i, err)
			continue
		}
		if req.Host != tt.out {
			t.Errorf("got %q; want %q", req.Host, tt.out)
		}
	}
}

func TestRequestInvalidMethod(t *testing.T) {
	_, err := NewRequest("bad method", "http://foo.com/", nil)
	if err == nil {
		t.Error("expected error from NewRequest with invalid method")
	}
	req, err := NewRequest("GET", "http://foo.example/", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Method = "bad method"
	_, err = DefaultClient.Do(req)
	if err == nil || !strings.Contains(err.Error(), "invalid method") {
		t.Errorf("Transport error = %v; want invalid method", err)
	}

	req, err = NewRequest("", "http://foo.com/", nil)
	if err != nil {
		t.Errorf("NewRequest(empty method) = %v; want nil", err)
	} else if req.Method != "GET" {
		t.Errorf("NewRequest(empty method) has method %q; want GET", req.Method)
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
		{strings.NewReader(""), 0},
		{NoBody, 0},

		// Not detected. During Go 1.8 we tried to make these set to -1, but
		// due to Issue 18117, we keep these returning 0, even though they're
		// unknown.
		{struct{ io.Reader }{strings.NewReader("xyz")}, 0},
		{io.NewSectionReader(strings.NewReader("x"), 0, 6), 0},
		{readByte(io.NewSectionReader(strings.NewReader("xy"), 0, 6)), 0},
	}
	for i, tt := range tests {
		req, err := NewRequest("POST", "http://localhost/", tt.r)
		if err != nil {
			t.Fatal(err)
		}
		if req.ContentLength != tt.want {
			t.Errorf("test[%d]: ContentLength(%T) = %d; want %d", i, tt.r, req.ContentLength, tt.want)
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

func TestRequestBadHost(t *testing.T) {
	got := []string{}
	req, err := NewRequest("GET", "http://foo/after", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Host = "foo.com with spaces"
	req.URL.Host = "foo.com with spaces"
	req.Write(logWrites{t, &got})
	want := []string{
		"GET /after HTTP/1.1\r\n",
		"Host: foo.com\r\n",
		"User-Agent: " + DefaultUserAgent + "\r\n",
		"\r\n",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Writes = %q\n  Want = %q", got, want)
	}
}

func TestStarRequest(t *testing.T) {
	req, err := ReadRequest(bufio.NewReader(strings.NewReader("M-SEARCH * HTTP/1.1\r\n\r\n")))
	if err != nil {
		return
	}
	if req.ContentLength != 0 {
		t.Errorf("ContentLength = %d; want 0", req.ContentLength)
	}
	if req.Body == nil {
		t.Errorf("Body = nil; want non-nil")
	}

	// Request.Write has Client semantics for Body/ContentLength,
	// where ContentLength 0 means unknown if Body is non-nil, and
	// thus chunking will happen unless we change semantics and
	// signal that we want to serialize it as exactly zero.  The
	// only way to do that for outbound requests is with a nil
	// Body:
	clientReq := *req
	clientReq.Body = nil

	var out bytes.Buffer
	if err := clientReq.Write(&out); err != nil {
		t.Fatal(err)
	}

	if strings.Contains(out.String(), "chunked") {
		t.Error("wrote chunked request; want no body")
	}
	back, err := ReadRequest(bufio.NewReader(bytes.NewReader(out.Bytes())))
	if err != nil {
		t.Fatal(err)
	}
	// Ignore the Headers (the User-Agent breaks the deep equal,
	// but we don't care about it)
	req.Header = nil
	back.Header = nil
	if !reflect.DeepEqual(req, back) {
		t.Errorf("Original request doesn't match Request read back.")
		t.Logf("Original: %#v", req)
		t.Logf("Original.URL: %#v", req.URL)
		t.Logf("Wrote: %s", out.Bytes())
		t.Logf("Read back (doesn't match Original): %#v", back)
	}
}

type responseWriterJustWriter struct {
	io.Writer
}

func (responseWriterJustWriter) Header() Header  { panic("should not be called") }
func (responseWriterJustWriter) WriteHeader(int) { panic("should not be called") }

// delayedEOFReader never returns (n > 0, io.EOF), instead putting
// off the io.EOF until a subsequent Read call.
type delayedEOFReader struct {
	r io.Reader
}

func (dr delayedEOFReader) Read(p []byte) (n int, err error) {
	n, err = dr.r.Read(p)
	if n > 0 && err == io.EOF {
		err = nil
	}
	return
}

func TestIssue10884_MaxBytesEOF(t *testing.T) {
	dst := ioutil.Discard
	_, err := io.Copy(dst, MaxBytesReader(
		responseWriterJustWriter{dst},
		ioutil.NopCloser(delayedEOFReader{strings.NewReader("12345")}),
		5))
	if err != nil {
		t.Fatal(err)
	}
}

// Issue 14981: MaxBytesReader's return error wasn't sticky. It
// doesn't technically need to be, but people expected it to be.
func TestMaxBytesReaderStickyError(t *testing.T) {
	isSticky := func(r io.Reader) error {
		var log bytes.Buffer
		buf := make([]byte, 1000)
		var firstErr error
		for {
			n, err := r.Read(buf)
			fmt.Fprintf(&log, "Read(%d) = %d, %v\n", len(buf), n, err)
			if err == nil {
				continue
			}
			if firstErr == nil {
				firstErr = err
				continue
			}
			if !reflect.DeepEqual(err, firstErr) {
				return fmt.Errorf("non-sticky error. got log:\n%s", log.Bytes())
			}
			t.Logf("Got log: %s", log.Bytes())
			return nil
		}
	}
	tests := [...]struct {
		readable int
		limit    int64
	}{
		0: {99, 100},
		1: {100, 100},
		2: {101, 100},
	}
	for i, tt := range tests {
		rc := MaxBytesReader(nil, ioutil.NopCloser(bytes.NewReader(make([]byte, tt.readable))), tt.limit)
		if err := isSticky(rc); err != nil {
			t.Errorf("%d. error: %v", i, err)
		}
	}
}

// verify that NewRequest sets Request.GetBody and that it works
func TestNewRequestGetBody(t *testing.T) {
	tests := []struct {
		r io.Reader
	}{
		{r: strings.NewReader("hello")},
		{r: bytes.NewReader([]byte("hello"))},
		{r: bytes.NewBuffer([]byte("hello"))},
	}
	for i, tt := range tests {
		req, err := NewRequest("POST", "http://foo.tld/", tt.r)
		if err != nil {
			t.Errorf("test[%d]: %v", i, err)
			continue
		}
		if req.Body == nil {
			t.Errorf("test[%d]: Body = nil", i)
			continue
		}
		if req.GetBody == nil {
			t.Errorf("test[%d]: GetBody = nil", i)
			continue
		}
		slurp1, err := ioutil.ReadAll(req.Body)
		if err != nil {
			t.Errorf("test[%d]: ReadAll(Body) = %v", i, err)
		}
		newBody, err := req.GetBody()
		if err != nil {
			t.Errorf("test[%d]: GetBody = %v", i, err)
		}
		slurp2, err := ioutil.ReadAll(newBody)
		if err != nil {
			t.Errorf("test[%d]: ReadAll(GetBody()) = %v", i, err)
		}
		if string(slurp1) != string(slurp2) {
			t.Errorf("test[%d]: Body %q != GetBody %q", i, slurp1, slurp2)
		}
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
