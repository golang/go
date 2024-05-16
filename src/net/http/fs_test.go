// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"errors"
	"fmt"
	"internal/testenv"
	"io"
	"io/fs"
	"mime"
	"mime/multipart"
	"net"
	"net/http"
	. "net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"testing/fstest"
	"time"
)

const (
	testFile    = "testdata/file"
	testFileLen = 11
)

type wantRange struct {
	start, end int64 // range [start,end)
}

var ServeFileRangeTests = []struct {
	r      string
	code   int
	ranges []wantRange
}{
	{r: "", code: StatusOK},
	{r: "bytes=0-4", code: StatusPartialContent, ranges: []wantRange{{0, 5}}},
	{r: "bytes=2-", code: StatusPartialContent, ranges: []wantRange{{2, testFileLen}}},
	{r: "bytes=-5", code: StatusPartialContent, ranges: []wantRange{{testFileLen - 5, testFileLen}}},
	{r: "bytes=3-7", code: StatusPartialContent, ranges: []wantRange{{3, 8}}},
	{r: "bytes=0-0,-2", code: StatusPartialContent, ranges: []wantRange{{0, 1}, {testFileLen - 2, testFileLen}}},
	{r: "bytes=0-1,5-8", code: StatusPartialContent, ranges: []wantRange{{0, 2}, {5, 9}}},
	{r: "bytes=0-1,5-", code: StatusPartialContent, ranges: []wantRange{{0, 2}, {5, testFileLen}}},
	{r: "bytes=5-1000", code: StatusPartialContent, ranges: []wantRange{{5, testFileLen}}},
	{r: "bytes=0-,1-,2-,3-,4-", code: StatusOK}, // ignore wasteful range request
	{r: "bytes=0-9", code: StatusPartialContent, ranges: []wantRange{{0, testFileLen - 1}}},
	{r: "bytes=0-10", code: StatusPartialContent, ranges: []wantRange{{0, testFileLen}}},
	{r: "bytes=0-11", code: StatusPartialContent, ranges: []wantRange{{0, testFileLen}}},
	{r: "bytes=10-11", code: StatusPartialContent, ranges: []wantRange{{testFileLen - 1, testFileLen}}},
	{r: "bytes=10-", code: StatusPartialContent, ranges: []wantRange{{testFileLen - 1, testFileLen}}},
	{r: "bytes=11-", code: StatusRequestedRangeNotSatisfiable},
	{r: "bytes=11-12", code: StatusRequestedRangeNotSatisfiable},
	{r: "bytes=12-12", code: StatusRequestedRangeNotSatisfiable},
	{r: "bytes=11-100", code: StatusRequestedRangeNotSatisfiable},
	{r: "bytes=12-100", code: StatusRequestedRangeNotSatisfiable},
	{r: "bytes=100-", code: StatusRequestedRangeNotSatisfiable},
	{r: "bytes=100-1000", code: StatusRequestedRangeNotSatisfiable},
}

func TestServeFile(t *testing.T) { run(t, testServeFile) }
func testServeFile(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		ServeFile(w, r, "testdata/file")
	})).ts
	c := ts.Client()

	var err error

	file, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatal("reading file:", err)
	}

	// set up the Request (re-used for all tests)
	var req Request
	req.Header = make(Header)
	if req.URL, err = url.Parse(ts.URL); err != nil {
		t.Fatal("ParseURL:", err)
	}

	// Get contents via various methods.
	//
	// See https://go.dev/issue/59471 for a proposal to limit the set of methods handled.
	// For now, test the historical behavior.
	for _, method := range []string{
		MethodGet,
		MethodPost,
		MethodPut,
		MethodPatch,
		MethodDelete,
		MethodOptions,
		MethodTrace,
	} {
		req.Method = method
		_, body := getBody(t, method, req, c)
		if !bytes.Equal(body, file) {
			t.Fatalf("body mismatch for %v request: got %q, want %q", method, body, file)
		}
	}

	// HEAD request.
	req.Method = MethodHead
	resp, body := getBody(t, "HEAD", req, c)
	if len(body) != 0 {
		t.Fatalf("body mismatch for HEAD request: got %q, want empty", body)
	}
	if got, want := resp.Header.Get("Content-Length"), fmt.Sprint(len(file)); got != want {
		t.Fatalf("Content-Length mismatch for HEAD request: got %v, want %v", got, want)
	}

	// Range tests
	req.Method = MethodGet
Cases:
	for _, rt := range ServeFileRangeTests {
		if rt.r != "" {
			req.Header.Set("Range", rt.r)
		}
		resp, body := getBody(t, fmt.Sprintf("range test %q", rt.r), req, c)
		if resp.StatusCode != rt.code {
			t.Errorf("range=%q: StatusCode=%d, want %d", rt.r, resp.StatusCode, rt.code)
		}
		if rt.code == StatusRequestedRangeNotSatisfiable {
			continue
		}
		wantContentRange := ""
		if len(rt.ranges) == 1 {
			rng := rt.ranges[0]
			wantContentRange = fmt.Sprintf("bytes %d-%d/%d", rng.start, rng.end-1, testFileLen)
		}
		cr := resp.Header.Get("Content-Range")
		if cr != wantContentRange {
			t.Errorf("range=%q: Content-Range = %q, want %q", rt.r, cr, wantContentRange)
		}
		ct := resp.Header.Get("Content-Type")
		if len(rt.ranges) == 1 {
			rng := rt.ranges[0]
			wantBody := file[rng.start:rng.end]
			if !bytes.Equal(body, wantBody) {
				t.Errorf("range=%q: body = %q, want %q", rt.r, body, wantBody)
			}
			if strings.HasPrefix(ct, "multipart/byteranges") {
				t.Errorf("range=%q content-type = %q; unexpected multipart/byteranges", rt.r, ct)
			}
		}
		if len(rt.ranges) > 1 {
			typ, params, err := mime.ParseMediaType(ct)
			if err != nil {
				t.Errorf("range=%q content-type = %q; %v", rt.r, ct, err)
				continue
			}
			if typ != "multipart/byteranges" {
				t.Errorf("range=%q content-type = %q; want multipart/byteranges", rt.r, typ)
				continue
			}
			if params["boundary"] == "" {
				t.Errorf("range=%q content-type = %q; lacks boundary", rt.r, ct)
				continue
			}
			if g, w := resp.ContentLength, int64(len(body)); g != w {
				t.Errorf("range=%q Content-Length = %d; want %d", rt.r, g, w)
				continue
			}
			mr := multipart.NewReader(bytes.NewReader(body), params["boundary"])
			for ri, rng := range rt.ranges {
				part, err := mr.NextPart()
				if err != nil {
					t.Errorf("range=%q, reading part index %d: %v", rt.r, ri, err)
					continue Cases
				}
				wantContentRange = fmt.Sprintf("bytes %d-%d/%d", rng.start, rng.end-1, testFileLen)
				if g, w := part.Header.Get("Content-Range"), wantContentRange; g != w {
					t.Errorf("range=%q: part Content-Range = %q; want %q", rt.r, g, w)
				}
				body, err := io.ReadAll(part)
				if err != nil {
					t.Errorf("range=%q, reading part index %d body: %v", rt.r, ri, err)
					continue Cases
				}
				wantBody := file[rng.start:rng.end]
				if !bytes.Equal(body, wantBody) {
					t.Errorf("range=%q: body = %q, want %q", rt.r, body, wantBody)
				}
			}
			_, err = mr.NextPart()
			if err != io.EOF {
				t.Errorf("range=%q; expected final error io.EOF; got %v", rt.r, err)
			}
		}
	}
}

func TestServeFile_DotDot(t *testing.T) {
	tests := []struct {
		req        string
		wantStatus int
	}{
		{"/testdata/file", 200},
		{"/../file", 400},
		{"/..", 400},
		{"/../", 400},
		{"/../foo", 400},
		{"/..\\foo", 400},
		{"/file/a", 200},
		{"/file/a..", 200},
		{"/file/a/..", 400},
		{"/file/a\\..", 400},
	}
	for _, tt := range tests {
		req, err := ReadRequest(bufio.NewReader(strings.NewReader("GET " + tt.req + " HTTP/1.1\r\nHost: foo\r\n\r\n")))
		if err != nil {
			t.Errorf("bad request %q: %v", tt.req, err)
			continue
		}
		rec := httptest.NewRecorder()
		ServeFile(rec, req, "testdata/file")
		if rec.Code != tt.wantStatus {
			t.Errorf("for request %q, status = %d; want %d", tt.req, rec.Code, tt.wantStatus)
		}
	}
}

// Tests that this doesn't panic. (Issue 30165)
func TestServeFileDirPanicEmptyPath(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/", nil)
	req.URL.Path = ""
	ServeFile(rec, req, "testdata")
	res := rec.Result()
	if res.StatusCode != 301 {
		t.Errorf("code = %v; want 301", res.Status)
	}
}

// Tests that ranges are ignored with serving empty content. (Issue 54794)
func TestServeContentWithEmptyContentIgnoreRanges(t *testing.T) {
	for _, r := range []string{
		"bytes=0-128",
		"bytes=1-",
	} {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest("GET", "/", nil)
		req.Header.Set("Range", r)
		ServeContent(rec, req, "nothing", time.Now(), bytes.NewReader(nil))
		res := rec.Result()
		if res.StatusCode != 200 {
			t.Errorf("code = %v; want 200", res.Status)
		}
		bodyLen := rec.Body.Len()
		if bodyLen != 0 {
			t.Errorf("body.Len() = %v; want 0", res.Status)
		}
	}
}

var fsRedirectTestData = []struct {
	original, redirect string
}{
	{"/test/index.html", "/test/"},
	{"/test/testdata", "/test/testdata/"},
	{"/test/testdata/file/", "/test/testdata/file"},
}

func TestFSRedirect(t *testing.T) { run(t, testFSRedirect) }
func testFSRedirect(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, StripPrefix("/test", FileServer(Dir(".")))).ts

	for _, data := range fsRedirectTestData {
		res, err := ts.Client().Get(ts.URL + data.original)
		if err != nil {
			t.Fatal(err)
		}
		res.Body.Close()
		if g, e := res.Request.URL.Path, data.redirect; g != e {
			t.Errorf("redirect from %s: got %s, want %s", data.original, g, e)
		}
	}
}

type testFileSystem struct {
	open func(name string) (File, error)
}

func (fs *testFileSystem) Open(name string) (File, error) {
	return fs.open(name)
}

func TestFileServerCleans(t *testing.T) {
	defer afterTest(t)
	ch := make(chan string, 1)
	fs := FileServer(&testFileSystem{func(name string) (File, error) {
		ch <- name
		return nil, errors.New("file does not exist")
	}})
	tests := []struct {
		reqPath, openArg string
	}{
		{"/foo.txt", "/foo.txt"},
		{"//foo.txt", "/foo.txt"},
		{"/../foo.txt", "/foo.txt"},
	}
	req, _ := NewRequest("GET", "http://example.com", nil)
	for n, test := range tests {
		rec := httptest.NewRecorder()
		req.URL.Path = test.reqPath
		fs.ServeHTTP(rec, req)
		if got := <-ch; got != test.openArg {
			t.Errorf("test %d: got %q, want %q", n, got, test.openArg)
		}
	}
}

func TestFileServerEscapesNames(t *testing.T) { run(t, testFileServerEscapesNames) }
func testFileServerEscapesNames(t *testing.T, mode testMode) {
	const dirListPrefix = "<!doctype html>\n<meta name=\"viewport\" content=\"width=device-width\">\n<pre>\n"
	const dirListSuffix = "\n</pre>\n"
	tests := []struct {
		name, escaped string
	}{
		{`simple_name`, `<a href="simple_name">simple_name</a>`},
		{`"'<>&`, `<a href="%22%27%3C%3E&">&#34;&#39;&lt;&gt;&amp;</a>`},
		{`?foo=bar#baz`, `<a href="%3Ffoo=bar%23baz">?foo=bar#baz</a>`},
		{`<combo>?foo`, `<a href="%3Ccombo%3E%3Ffoo">&lt;combo&gt;?foo</a>`},
		{`foo:bar`, `<a href="./foo:bar">foo:bar</a>`},
	}

	// We put each test file in its own directory in the fakeFS so we can look at it in isolation.
	fs := make(fakeFS)
	for i, test := range tests {
		testFile := &fakeFileInfo{basename: test.name}
		fs[fmt.Sprintf("/%d", i)] = &fakeFileInfo{
			dir:     true,
			modtime: time.Unix(1000000000, 0).UTC(),
			ents:    []*fakeFileInfo{testFile},
		}
		fs[fmt.Sprintf("/%d/%s", i, test.name)] = testFile
	}

	ts := newClientServerTest(t, mode, FileServer(&fs)).ts
	for i, test := range tests {
		url := fmt.Sprintf("%s/%d", ts.URL, i)
		res, err := ts.Client().Get(url)
		if err != nil {
			t.Fatalf("test %q: Get: %v", test.name, err)
		}
		b, err := io.ReadAll(res.Body)
		if err != nil {
			t.Fatalf("test %q: read Body: %v", test.name, err)
		}
		s := string(b)
		if !strings.HasPrefix(s, dirListPrefix) || !strings.HasSuffix(s, dirListSuffix) {
			t.Errorf("test %q: listing dir, full output is %q, want prefix %q and suffix %q", test.name, s, dirListPrefix, dirListSuffix)
		}
		if trimmed := strings.TrimSuffix(strings.TrimPrefix(s, dirListPrefix), dirListSuffix); trimmed != test.escaped {
			t.Errorf("test %q: listing dir, filename escaped to %q, want %q", test.name, trimmed, test.escaped)
		}
		res.Body.Close()
	}
}

func TestFileServerSortsNames(t *testing.T) { run(t, testFileServerSortsNames) }
func testFileServerSortsNames(t *testing.T, mode testMode) {
	const contents = "I am a fake file"
	dirMod := time.Unix(123, 0).UTC()
	fileMod := time.Unix(1000000000, 0).UTC()
	fs := fakeFS{
		"/": &fakeFileInfo{
			dir:     true,
			modtime: dirMod,
			ents: []*fakeFileInfo{
				{
					basename: "b",
					modtime:  fileMod,
					contents: contents,
				},
				{
					basename: "a",
					modtime:  fileMod,
					contents: contents,
				},
			},
		},
	}

	ts := newClientServerTest(t, mode, FileServer(&fs)).ts

	res, err := ts.Client().Get(ts.URL)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	defer res.Body.Close()

	b, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatalf("read Body: %v", err)
	}
	s := string(b)
	if !strings.Contains(s, "<a href=\"a\">a</a>\n<a href=\"b\">b</a>") {
		t.Errorf("output appears to be unsorted:\n%s", s)
	}
}

func mustRemoveAll(dir string) {
	err := os.RemoveAll(dir)
	if err != nil {
		panic(err)
	}
}

func TestFileServerImplicitLeadingSlash(t *testing.T) { run(t, testFileServerImplicitLeadingSlash) }
func testFileServerImplicitLeadingSlash(t *testing.T, mode testMode) {
	tempDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(tempDir, "foo.txt"), []byte("Hello world"), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	ts := newClientServerTest(t, mode, StripPrefix("/bar/", FileServer(Dir(tempDir)))).ts
	get := func(suffix string) string {
		res, err := ts.Client().Get(ts.URL + suffix)
		if err != nil {
			t.Fatalf("Get %s: %v", suffix, err)
		}
		b, err := io.ReadAll(res.Body)
		if err != nil {
			t.Fatalf("ReadAll %s: %v", suffix, err)
		}
		res.Body.Close()
		return string(b)
	}
	if s := get("/bar/"); !strings.Contains(s, ">foo.txt<") {
		t.Logf("expected a directory listing with foo.txt, got %q", s)
	}
	if s := get("/bar/foo.txt"); s != "Hello world" {
		t.Logf("expected %q, got %q", "Hello world", s)
	}
}

func TestDirJoin(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("skipping test on windows")
	}
	wfi, err := os.Stat("/etc/hosts")
	if err != nil {
		t.Skip("skipping test; no /etc/hosts file")
	}
	test := func(d Dir, name string) {
		f, err := d.Open(name)
		if err != nil {
			t.Fatalf("open of %s: %v", name, err)
		}
		defer f.Close()
		gfi, err := f.Stat()
		if err != nil {
			t.Fatalf("stat of %s: %v", name, err)
		}
		if !os.SameFile(gfi, wfi) {
			t.Errorf("%s got different file", name)
		}
	}
	test(Dir("/etc/"), "/hosts")
	test(Dir("/etc/"), "hosts")
	test(Dir("/etc/"), "../../../../hosts")
	test(Dir("/etc"), "/hosts")
	test(Dir("/etc"), "hosts")
	test(Dir("/etc"), "../../../../hosts")

	// Not really directories, but since we use this trick in
	// ServeFile, test it:
	test(Dir("/etc/hosts"), "")
	test(Dir("/etc/hosts"), "/")
	test(Dir("/etc/hosts"), "../")
}

func TestEmptyDirOpenCWD(t *testing.T) {
	test := func(d Dir) {
		name := "fs_test.go"
		f, err := d.Open(name)
		if err != nil {
			t.Fatalf("open of %s: %v", name, err)
		}
		defer f.Close()
	}
	test(Dir(""))
	test(Dir("."))
	test(Dir("./"))
}

func TestServeFileContentType(t *testing.T) { run(t, testServeFileContentType) }
func testServeFileContentType(t *testing.T, mode testMode) {
	const ctype = "icecream/chocolate"
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		switch r.FormValue("override") {
		case "1":
			w.Header().Set("Content-Type", ctype)
		case "2":
			// Explicitly inhibit sniffing.
			w.Header()["Content-Type"] = []string{}
		}
		ServeFile(w, r, "testdata/file")
	})).ts
	get := func(override string, want []string) {
		resp, err := ts.Client().Get(ts.URL + "?override=" + override)
		if err != nil {
			t.Fatal(err)
		}
		if h := resp.Header["Content-Type"]; !reflect.DeepEqual(h, want) {
			t.Errorf("Content-Type mismatch: got %v, want %v", h, want)
		}
		resp.Body.Close()
	}
	get("0", []string{"text/plain; charset=utf-8"})
	get("1", []string{ctype})
	get("2", nil)
}

func TestServeFileMimeType(t *testing.T) { run(t, testServeFileMimeType) }
func testServeFileMimeType(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		ServeFile(w, r, "testdata/style.css")
	})).ts
	resp, err := ts.Client().Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	want := "text/css; charset=utf-8"
	if h := resp.Header.Get("Content-Type"); h != want {
		t.Errorf("Content-Type mismatch: got %q, want %q", h, want)
	}
}

func TestServeFileFromCWD(t *testing.T) { run(t, testServeFileFromCWD) }
func testServeFileFromCWD(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		ServeFile(w, r, "fs_test.go")
	})).ts
	r, err := ts.Client().Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	if r.StatusCode != 200 {
		t.Fatalf("expected 200 OK, got %s", r.Status)
	}
}

// Issue 13996
func TestServeDirWithoutTrailingSlash(t *testing.T) { run(t, testServeDirWithoutTrailingSlash) }
func testServeDirWithoutTrailingSlash(t *testing.T, mode testMode) {
	e := "/testdata/"
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		ServeFile(w, r, ".")
	})).ts
	r, err := ts.Client().Get(ts.URL + "/testdata")
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	if g := r.Request.URL.Path; g != e {
		t.Errorf("got %s, want %s", g, e)
	}
}

// Tests that ServeFile doesn't add a Content-Length if a Content-Encoding is
// specified.
func TestServeFileWithContentEncoding(t *testing.T) { run(t, testServeFileWithContentEncoding) }
func testServeFileWithContentEncoding(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Encoding", "foo")
		ServeFile(w, r, "testdata/file")

		// Because the testdata is so small, it would fit in
		// both the h1 and h2 Server's write buffers. For h1,
		// sendfile is used, though, forcing a header flush at
		// the io.Copy. http2 doesn't do a header flush so
		// buffers all 11 bytes and then adds its own
		// Content-Length. To prevent the Server's
		// Content-Length and test ServeFile only, flush here.
		w.(Flusher).Flush()
	}))
	resp, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if g, e := resp.ContentLength, int64(-1); g != e {
		t.Errorf("Content-Length mismatch: got %d, want %d", g, e)
	}
}

// Tests that ServeFile does not generate representation metadata when
// file has not been modified, as per RFC 7232 section 4.1.
func TestServeFileNotModified(t *testing.T) { run(t, testServeFileNotModified) }
func testServeFileNotModified(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Content-Encoding", "foo")
		w.Header().Set("Etag", `"123"`)
		ServeFile(w, r, "testdata/file")

		// Because the testdata is so small, it would fit in
		// both the h1 and h2 Server's write buffers. For h1,
		// sendfile is used, though, forcing a header flush at
		// the io.Copy. http2 doesn't do a header flush so
		// buffers all 11 bytes and then adds its own
		// Content-Length. To prevent the Server's
		// Content-Length and test ServeFile only, flush here.
		w.(Flusher).Flush()
	}))
	req, err := NewRequest("GET", cst.ts.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("If-None-Match", `"123"`)
	resp, err := cst.c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	b, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		t.Fatal("reading Body:", err)
	}
	if len(b) != 0 {
		t.Errorf("non-empty body")
	}
	if g, e := resp.StatusCode, StatusNotModified; g != e {
		t.Errorf("status mismatch: got %d, want %d", g, e)
	}
	// HTTP1 transport sets ContentLength to 0.
	if g, e1, e2 := resp.ContentLength, int64(-1), int64(0); g != e1 && g != e2 {
		t.Errorf("Content-Length mismatch: got %d, want %d or %d", g, e1, e2)
	}
	if resp.Header.Get("Content-Type") != "" {
		t.Errorf("Content-Type present, but it should not be")
	}
	if resp.Header.Get("Content-Encoding") != "" {
		t.Errorf("Content-Encoding present, but it should not be")
	}
}

func TestServeIndexHtml(t *testing.T) { run(t, testServeIndexHtml) }
func testServeIndexHtml(t *testing.T, mode testMode) {
	for i := 0; i < 2; i++ {
		var h Handler
		var name string
		switch i {
		case 0:
			h = FileServer(Dir("."))
			name = "Dir"
		case 1:
			h = FileServer(FS(os.DirFS(".")))
			name = "DirFS"
		}
		t.Run(name, func(t *testing.T) {
			const want = "index.html says hello\n"
			ts := newClientServerTest(t, mode, h).ts

			for _, path := range []string{"/testdata/", "/testdata/index.html"} {
				res, err := ts.Client().Get(ts.URL + path)
				if err != nil {
					t.Fatal(err)
				}
				b, err := io.ReadAll(res.Body)
				if err != nil {
					t.Fatal("reading Body:", err)
				}
				if s := string(b); s != want {
					t.Errorf("for path %q got %q, want %q", path, s, want)
				}
				res.Body.Close()
			}
		})
	}
}

func TestServeIndexHtmlFS(t *testing.T) { run(t, testServeIndexHtmlFS) }
func testServeIndexHtmlFS(t *testing.T, mode testMode) {
	const want = "index.html says hello\n"
	ts := newClientServerTest(t, mode, FileServer(Dir("."))).ts
	defer ts.Close()

	for _, path := range []string{"/testdata/", "/testdata/index.html"} {
		res, err := ts.Client().Get(ts.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		b, err := io.ReadAll(res.Body)
		if err != nil {
			t.Fatal("reading Body:", err)
		}
		if s := string(b); s != want {
			t.Errorf("for path %q got %q, want %q", path, s, want)
		}
		res.Body.Close()
	}
}

func TestFileServerZeroByte(t *testing.T) { run(t, testFileServerZeroByte) }
func testFileServerZeroByte(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, FileServer(Dir("."))).ts

	c, err := net.Dial("tcp", ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	_, err = fmt.Fprintf(c, "GET /..\x00 HTTP/1.0\r\n\r\n")
	if err != nil {
		t.Fatal(err)
	}
	var got bytes.Buffer
	bufr := bufio.NewReader(io.TeeReader(c, &got))
	res, err := ReadResponse(bufr, nil)
	if err != nil {
		t.Fatal("ReadResponse: ", err)
	}
	if res.StatusCode == 200 {
		t.Errorf("got status 200; want an error. Body is:\n%s", got.Bytes())
	}
}

func TestFileServerNamesEscape(t *testing.T) { run(t, testFileServerNamesEscape) }
func testFileServerNamesEscape(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, FileServer(Dir("testdata"))).ts
	for _, path := range []string{
		"/../testdata/file",
		"/NUL", // don't read from device files on Windows
	} {
		res, err := ts.Client().Get(ts.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		res.Body.Close()
		if res.StatusCode < 400 || res.StatusCode > 599 {
			t.Errorf("Get(%q): got status %v, want 4xx or 5xx", path, res.StatusCode)
		}

	}
}

type fakeFileInfo struct {
	dir      bool
	basename string
	modtime  time.Time
	ents     []*fakeFileInfo
	contents string
	err      error
}

func (f *fakeFileInfo) Name() string       { return f.basename }
func (f *fakeFileInfo) Sys() any           { return nil }
func (f *fakeFileInfo) ModTime() time.Time { return f.modtime }
func (f *fakeFileInfo) IsDir() bool        { return f.dir }
func (f *fakeFileInfo) Size() int64        { return int64(len(f.contents)) }
func (f *fakeFileInfo) Mode() fs.FileMode {
	if f.dir {
		return 0755 | fs.ModeDir
	}
	return 0644
}

func (f *fakeFileInfo) String() string {
	return fs.FormatFileInfo(f)
}

type fakeFile struct {
	io.ReadSeeker
	fi     *fakeFileInfo
	path   string // as opened
	entpos int
}

func (f *fakeFile) Close() error               { return nil }
func (f *fakeFile) Stat() (fs.FileInfo, error) { return f.fi, nil }
func (f *fakeFile) Readdir(count int) ([]fs.FileInfo, error) {
	if !f.fi.dir {
		return nil, fs.ErrInvalid
	}
	var fis []fs.FileInfo

	limit := f.entpos + count
	if count <= 0 || limit > len(f.fi.ents) {
		limit = len(f.fi.ents)
	}
	for ; f.entpos < limit; f.entpos++ {
		fis = append(fis, f.fi.ents[f.entpos])
	}

	if len(fis) == 0 && count > 0 {
		return fis, io.EOF
	} else {
		return fis, nil
	}
}

type fakeFS map[string]*fakeFileInfo

func (fsys fakeFS) Open(name string) (File, error) {
	name = path.Clean(name)
	f, ok := fsys[name]
	if !ok {
		return nil, fs.ErrNotExist
	}
	if f.err != nil {
		return nil, f.err
	}
	return &fakeFile{ReadSeeker: strings.NewReader(f.contents), fi: f, path: name}, nil
}

func TestDirectoryIfNotModified(t *testing.T) { run(t, testDirectoryIfNotModified) }
func testDirectoryIfNotModified(t *testing.T, mode testMode) {
	const indexContents = "I am a fake index.html file"
	fileMod := time.Unix(1000000000, 0).UTC()
	fileModStr := fileMod.Format(TimeFormat)
	dirMod := time.Unix(123, 0).UTC()
	indexFile := &fakeFileInfo{
		basename: "index.html",
		modtime:  fileMod,
		contents: indexContents,
	}
	fs := fakeFS{
		"/": &fakeFileInfo{
			dir:     true,
			modtime: dirMod,
			ents:    []*fakeFileInfo{indexFile},
		},
		"/index.html": indexFile,
	}

	ts := newClientServerTest(t, mode, FileServer(fs)).ts

	res, err := ts.Client().Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	b, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != indexContents {
		t.Fatalf("Got body %q; want %q", b, indexContents)
	}
	res.Body.Close()

	lastMod := res.Header.Get("Last-Modified")
	if lastMod != fileModStr {
		t.Fatalf("initial Last-Modified = %q; want %q", lastMod, fileModStr)
	}

	req, _ := NewRequest("GET", ts.URL, nil)
	req.Header.Set("If-Modified-Since", lastMod)

	c := ts.Client()
	res, err = c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != 304 {
		t.Fatalf("Code after If-Modified-Since request = %v; want 304", res.StatusCode)
	}
	res.Body.Close()

	// Advance the index.html file's modtime, but not the directory's.
	indexFile.modtime = indexFile.modtime.Add(1 * time.Hour)

	res, err = c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != 200 {
		t.Fatalf("Code after second If-Modified-Since request = %v; want 200; res is %#v", res.StatusCode, res)
	}
	res.Body.Close()
}

func mustStat(t *testing.T, fileName string) fs.FileInfo {
	fi, err := os.Stat(fileName)
	if err != nil {
		t.Fatal(err)
	}
	return fi
}

func TestServeContent(t *testing.T) { run(t, testServeContent) }
func testServeContent(t *testing.T, mode testMode) {
	type serveParam struct {
		name        string
		modtime     time.Time
		content     io.ReadSeeker
		contentType string
		etag        string
	}
	servec := make(chan serveParam, 1)
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		p := <-servec
		if p.etag != "" {
			w.Header().Set("ETag", p.etag)
		}
		if p.contentType != "" {
			w.Header().Set("Content-Type", p.contentType)
		}
		ServeContent(w, r, p.name, p.modtime, p.content)
	})).ts

	type testCase struct {
		// One of file or content must be set:
		file    string
		content io.ReadSeeker

		modtime          time.Time
		serveETag        string // optional
		serveContentType string // optional
		reqHeader        map[string]string
		wantLastMod      string
		wantContentType  string
		wantContentRange string
		wantStatus       int
	}
	htmlModTime := mustStat(t, "testdata/index.html").ModTime()
	tests := map[string]testCase{
		"no_last_modified": {
			file:            "testdata/style.css",
			wantContentType: "text/css; charset=utf-8",
			wantStatus:      200,
		},
		"with_last_modified": {
			file:            "testdata/index.html",
			wantContentType: "text/html; charset=utf-8",
			modtime:         htmlModTime,
			wantLastMod:     htmlModTime.UTC().Format(TimeFormat),
			wantStatus:      200,
		},
		"not_modified_modtime": {
			file:      "testdata/style.css",
			serveETag: `"foo"`, // Last-Modified sent only when no ETag
			modtime:   htmlModTime,
			reqHeader: map[string]string{
				"If-Modified-Since": htmlModTime.UTC().Format(TimeFormat),
			},
			wantStatus: 304,
		},
		"not_modified_modtime_with_contenttype": {
			file:             "testdata/style.css",
			serveContentType: "text/css", // explicit content type
			serveETag:        `"foo"`,    // Last-Modified sent only when no ETag
			modtime:          htmlModTime,
			reqHeader: map[string]string{
				"If-Modified-Since": htmlModTime.UTC().Format(TimeFormat),
			},
			wantStatus: 304,
		},
		"not_modified_etag": {
			file:      "testdata/style.css",
			serveETag: `"foo"`,
			reqHeader: map[string]string{
				"If-None-Match": `"foo"`,
			},
			wantStatus: 304,
		},
		"not_modified_etag_no_seek": {
			content:   panicOnSeek{nil}, // should never be called
			serveETag: `W/"foo"`,        // If-None-Match uses weak ETag comparison
			reqHeader: map[string]string{
				"If-None-Match": `"baz", W/"foo"`,
			},
			wantStatus: 304,
		},
		"if_none_match_mismatch": {
			file:      "testdata/style.css",
			serveETag: `"foo"`,
			reqHeader: map[string]string{
				"If-None-Match": `"Foo"`,
			},
			wantStatus:      200,
			wantContentType: "text/css; charset=utf-8",
		},
		"if_none_match_malformed": {
			file:      "testdata/style.css",
			serveETag: `"foo"`,
			reqHeader: map[string]string{
				"If-None-Match": `,`,
			},
			wantStatus:      200,
			wantContentType: "text/css; charset=utf-8",
		},
		"range_good": {
			file:      "testdata/style.css",
			serveETag: `"A"`,
			reqHeader: map[string]string{
				"Range": "bytes=0-4",
			},
			wantStatus:       StatusPartialContent,
			wantContentType:  "text/css; charset=utf-8",
			wantContentRange: "bytes 0-4/8",
		},
		"range_match": {
			file:      "testdata/style.css",
			serveETag: `"A"`,
			reqHeader: map[string]string{
				"Range":    "bytes=0-4",
				"If-Range": `"A"`,
			},
			wantStatus:       StatusPartialContent,
			wantContentType:  "text/css; charset=utf-8",
			wantContentRange: "bytes 0-4/8",
		},
		"range_match_weak_etag": {
			file:      "testdata/style.css",
			serveETag: `W/"A"`,
			reqHeader: map[string]string{
				"Range":    "bytes=0-4",
				"If-Range": `W/"A"`,
			},
			wantStatus:      200,
			wantContentType: "text/css; charset=utf-8",
		},
		"range_no_overlap": {
			file:      "testdata/style.css",
			serveETag: `"A"`,
			reqHeader: map[string]string{
				"Range": "bytes=10-20",
			},
			wantStatus:       StatusRequestedRangeNotSatisfiable,
			wantContentType:  "text/plain; charset=utf-8",
			wantContentRange: "bytes */8",
		},
		// An If-Range resource for entity "A", but entity "B" is now current.
		// The Range request should be ignored.
		"range_no_match": {
			file:      "testdata/style.css",
			serveETag: `"A"`,
			reqHeader: map[string]string{
				"Range":    "bytes=0-4",
				"If-Range": `"B"`,
			},
			wantStatus:      200,
			wantContentType: "text/css; charset=utf-8",
		},
		"range_with_modtime": {
			file:    "testdata/style.css",
			modtime: time.Date(2014, 6, 25, 17, 12, 18, 0 /* nanos */, time.UTC),
			reqHeader: map[string]string{
				"Range":    "bytes=0-4",
				"If-Range": "Wed, 25 Jun 2014 17:12:18 GMT",
			},
			wantStatus:       StatusPartialContent,
			wantContentType:  "text/css; charset=utf-8",
			wantContentRange: "bytes 0-4/8",
			wantLastMod:      "Wed, 25 Jun 2014 17:12:18 GMT",
		},
		"range_with_modtime_mismatch": {
			file:    "testdata/style.css",
			modtime: time.Date(2014, 6, 25, 17, 12, 18, 0 /* nanos */, time.UTC),
			reqHeader: map[string]string{
				"Range":    "bytes=0-4",
				"If-Range": "Wed, 25 Jun 2014 17:12:19 GMT",
			},
			wantStatus:      StatusOK,
			wantContentType: "text/css; charset=utf-8",
			wantLastMod:     "Wed, 25 Jun 2014 17:12:18 GMT",
		},
		"range_with_modtime_nanos": {
			file:    "testdata/style.css",
			modtime: time.Date(2014, 6, 25, 17, 12, 18, 123 /* nanos */, time.UTC),
			reqHeader: map[string]string{
				"Range":    "bytes=0-4",
				"If-Range": "Wed, 25 Jun 2014 17:12:18 GMT",
			},
			wantStatus:       StatusPartialContent,
			wantContentType:  "text/css; charset=utf-8",
			wantContentRange: "bytes 0-4/8",
			wantLastMod:      "Wed, 25 Jun 2014 17:12:18 GMT",
		},
		"unix_zero_modtime": {
			content:         strings.NewReader("<html>foo"),
			modtime:         time.Unix(0, 0),
			wantStatus:      StatusOK,
			wantContentType: "text/html; charset=utf-8",
		},
		"ifmatch_matches": {
			file:      "testdata/style.css",
			serveETag: `"A"`,
			reqHeader: map[string]string{
				"If-Match": `"Z", "A"`,
			},
			wantStatus:      200,
			wantContentType: "text/css; charset=utf-8",
		},
		"ifmatch_star": {
			file:      "testdata/style.css",
			serveETag: `"A"`,
			reqHeader: map[string]string{
				"If-Match": `*`,
			},
			wantStatus:      200,
			wantContentType: "text/css; charset=utf-8",
		},
		"ifmatch_failed": {
			file:      "testdata/style.css",
			serveETag: `"A"`,
			reqHeader: map[string]string{
				"If-Match": `"B"`,
			},
			wantStatus: 412,
		},
		"ifmatch_fails_on_weak_etag": {
			file:      "testdata/style.css",
			serveETag: `W/"A"`,
			reqHeader: map[string]string{
				"If-Match": `W/"A"`,
			},
			wantStatus: 412,
		},
		"if_unmodified_since_true": {
			file:    "testdata/style.css",
			modtime: htmlModTime,
			reqHeader: map[string]string{
				"If-Unmodified-Since": htmlModTime.UTC().Format(TimeFormat),
			},
			wantStatus:      200,
			wantContentType: "text/css; charset=utf-8",
			wantLastMod:     htmlModTime.UTC().Format(TimeFormat),
		},
		"if_unmodified_since_false": {
			file:    "testdata/style.css",
			modtime: htmlModTime,
			reqHeader: map[string]string{
				"If-Unmodified-Since": htmlModTime.Add(-2 * time.Second).UTC().Format(TimeFormat),
			},
			wantStatus:  412,
			wantLastMod: htmlModTime.UTC().Format(TimeFormat),
		},
	}
	for testName, tt := range tests {
		var content io.ReadSeeker
		if tt.file != "" {
			f, err := os.Open(tt.file)
			if err != nil {
				t.Fatalf("test %q: %v", testName, err)
			}
			defer f.Close()
			content = f
		} else {
			content = tt.content
		}
		for _, method := range []string{"GET", "HEAD"} {
			//restore content in case it is consumed by previous method
			if content, ok := content.(*strings.Reader); ok {
				content.Seek(0, io.SeekStart)
			}

			servec <- serveParam{
				name:        filepath.Base(tt.file),
				content:     content,
				modtime:     tt.modtime,
				etag:        tt.serveETag,
				contentType: tt.serveContentType,
			}
			req, err := NewRequest(method, ts.URL, nil)
			if err != nil {
				t.Fatal(err)
			}
			for k, v := range tt.reqHeader {
				req.Header.Set(k, v)
			}

			c := ts.Client()
			res, err := c.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			io.Copy(io.Discard, res.Body)
			res.Body.Close()
			if res.StatusCode != tt.wantStatus {
				t.Errorf("test %q using %q: got status = %d; want %d", testName, method, res.StatusCode, tt.wantStatus)
			}
			if g, e := res.Header.Get("Content-Type"), tt.wantContentType; g != e {
				t.Errorf("test %q using %q: got content-type = %q, want %q", testName, method, g, e)
			}
			if g, e := res.Header.Get("Content-Range"), tt.wantContentRange; g != e {
				t.Errorf("test %q using %q: got content-range = %q, want %q", testName, method, g, e)
			}
			if g, e := res.Header.Get("Last-Modified"), tt.wantLastMod; g != e {
				t.Errorf("test %q using %q: got last-modified = %q, want %q", testName, method, g, e)
			}
		}
	}
}

// Issue 12991
func TestServerFileStatError(t *testing.T) {
	rec := httptest.NewRecorder()
	r, _ := NewRequest("GET", "http://foo/", nil)
	redirect := false
	name := "file.txt"
	fs := issue12991FS{}
	ExportServeFile(rec, r, fs, name, redirect)
	if body := rec.Body.String(); !strings.Contains(body, "403") || !strings.Contains(body, "Forbidden") {
		t.Errorf("wanted 403 forbidden message; got: %s", body)
	}
}

type issue12991FS struct{}

func (issue12991FS) Open(string) (File, error) { return issue12991File{}, nil }

type issue12991File struct{ File }

func (issue12991File) Stat() (fs.FileInfo, error) { return nil, fs.ErrPermission }
func (issue12991File) Close() error               { return nil }

func TestFileServerErrorMessages(t *testing.T) { run(t, testFileServerErrorMessages) }
func testFileServerErrorMessages(t *testing.T, mode testMode) {
	fs := fakeFS{
		"/500": &fakeFileInfo{
			err: errors.New("random error"),
		},
		"/403": &fakeFileInfo{
			err: &fs.PathError{Err: fs.ErrPermission},
		},
	}
	server := FileServer(fs)
	h := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Etag", "étude")
		w.Header().Set("Cache-Control", "yes")
		w.Header().Set("Content-Type", "awesome")
		w.Header().Set("Last-Modified", "yesterday")
		server.ServeHTTP(w, r)
	}
	ts := newClientServerTest(t, mode, http.HandlerFunc(h)).ts
	c := ts.Client()
	for _, code := range []int{403, 404, 500} {
		res, err := c.Get(fmt.Sprintf("%s/%d", ts.URL, code))
		if err != nil {
			t.Errorf("Error fetching /%d: %v", code, err)
			continue
		}
		res.Body.Close()
		if res.StatusCode != code {
			t.Errorf("GET /%d: StatusCode = %d; want %d", code, res.StatusCode, code)
		}
		for _, hdr := range []string{"Etag", "Last-Modified", "Cache-Control"} {
			if v, ok := res.Header[hdr]; ok {
				t.Errorf("GET /%d: Header[%q] = %q, want not present", code, hdr, v)
			}
		}
	}
}

// verifies that sendfile is being used on Linux
func TestLinuxSendfile(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	if runtime.GOOS != "linux" {
		t.Skip("skipping; linux-only test")
	}
	if _, err := exec.LookPath("strace"); err != nil {
		t.Skip("skipping; strace not found in path")
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	lnf, err := ln.(*net.TCPListener).File()
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	// Attempt to run strace, and skip on failure - this test requires SYS_PTRACE.
	if err := testenv.Command(t, "strace", "-f", "-q", os.Args[0], "-test.run=^$").Run(); err != nil {
		t.Skipf("skipping; failed to run strace: %v", err)
	}

	filename := fmt.Sprintf("1kb-%d", os.Getpid())
	filepath := path.Join(os.TempDir(), filename)

	if err := os.WriteFile(filepath, bytes.Repeat([]byte{'a'}, 1<<10), 0755); err != nil {
		t.Fatal(err)
	}
	defer os.Remove(filepath)

	var buf strings.Builder
	child := testenv.Command(t, "strace", "-f", "-q", os.Args[0], "-test.run=^TestLinuxSendfileChild$")
	child.ExtraFiles = append(child.ExtraFiles, lnf)
	child.Env = append([]string{"GO_WANT_HELPER_PROCESS=1"}, os.Environ()...)
	child.Stdout = &buf
	child.Stderr = &buf
	if err := child.Start(); err != nil {
		t.Skipf("skipping; failed to start straced child: %v", err)
	}

	res, err := Get(fmt.Sprintf("http://%s/%s", ln.Addr(), filename))
	if err != nil {
		t.Fatalf("http client error: %v", err)
	}
	_, err = io.Copy(io.Discard, res.Body)
	if err != nil {
		t.Fatalf("client body read error: %v", err)
	}
	res.Body.Close()

	// Force child to exit cleanly.
	Post(fmt.Sprintf("http://%s/quit", ln.Addr()), "", nil)
	child.Wait()

	rx := regexp.MustCompile(`\b(n64:)?sendfile(64)?\(`)
	out := buf.String()
	if !rx.MatchString(out) {
		t.Errorf("no sendfile system call found in:\n%s", out)
	}
}

func getBody(t *testing.T, testName string, req Request, client *Client) (*Response, []byte) {
	r, err := client.Do(&req)
	if err != nil {
		t.Fatalf("%s: for URL %q, send error: %v", testName, req.URL.String(), err)
	}
	b, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("%s: for URL %q, reading body: %v", testName, req.URL.String(), err)
	}
	return r, b
}

// TestLinuxSendfileChild isn't a real test. It's used as a helper process
// for TestLinuxSendfile.
func TestLinuxSendfileChild(*testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	defer os.Exit(0)
	fd3 := os.NewFile(3, "ephemeral-port-listener")
	ln, err := net.FileListener(fd3)
	if err != nil {
		panic(err)
	}
	mux := NewServeMux()
	mux.Handle("/", FileServer(Dir(os.TempDir())))
	mux.HandleFunc("/quit", func(ResponseWriter, *Request) {
		os.Exit(0)
	})
	s := &Server{Handler: mux}
	err = s.Serve(ln)
	if err != nil {
		panic(err)
	}
}

// Issues 18984, 49552: tests that requests for paths beyond files return not-found errors
func TestFileServerNotDirError(t *testing.T) {
	run(t, func(t *testing.T, mode testMode) {
		t.Run("Dir", func(t *testing.T) {
			testFileServerNotDirError(t, mode, func(path string) FileSystem { return Dir(path) })
		})
		t.Run("FS", func(t *testing.T) {
			testFileServerNotDirError(t, mode, func(path string) FileSystem { return FS(os.DirFS(path)) })
		})
	})
}

func testFileServerNotDirError(t *testing.T, mode testMode, newfs func(string) FileSystem) {
	ts := newClientServerTest(t, mode, FileServer(newfs("testdata"))).ts

	res, err := ts.Client().Get(ts.URL + "/index.html/not-a-file")
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.StatusCode != 404 {
		t.Errorf("StatusCode = %v; want 404", res.StatusCode)
	}

	test := func(name string, fsys FileSystem) {
		t.Run(name, func(t *testing.T) {
			_, err = fsys.Open("/index.html/not-a-file")
			if err == nil {
				t.Fatal("err == nil; want != nil")
			}
			if !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("err = %v; errors.Is(err, fs.ErrNotExist) = %v; want true", err,
					errors.Is(err, fs.ErrNotExist))
			}

			_, err = fsys.Open("/index.html/not-a-dir/not-a-file")
			if err == nil {
				t.Fatal("err == nil; want != nil")
			}
			if !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("err = %v; errors.Is(err, fs.ErrNotExist) = %v; want true", err,
					errors.Is(err, fs.ErrNotExist))
			}
		})
	}

	absPath, err := filepath.Abs("testdata")
	if err != nil {
		t.Fatal("get abs path:", err)
	}

	test("RelativePath", newfs("testdata"))
	test("AbsolutePath", newfs(absPath))
}

func TestFileServerCleanPath(t *testing.T) {
	tests := []struct {
		path     string
		wantCode int
		wantOpen []string
	}{
		{"/", 200, []string{"/", "/index.html"}},
		{"/dir", 301, []string{"/dir"}},
		{"/dir/", 200, []string{"/dir", "/dir/index.html"}},
	}
	for _, tt := range tests {
		var log []string
		rr := httptest.NewRecorder()
		req, _ := NewRequest("GET", "http://foo.localhost"+tt.path, nil)
		FileServer(fileServerCleanPathDir{&log}).ServeHTTP(rr, req)
		if !reflect.DeepEqual(log, tt.wantOpen) {
			t.Logf("For %s: Opens = %q; want %q", tt.path, log, tt.wantOpen)
		}
		if rr.Code != tt.wantCode {
			t.Logf("For %s: Response code = %d; want %d", tt.path, rr.Code, tt.wantCode)
		}
	}
}

type fileServerCleanPathDir struct {
	log *[]string
}

func (d fileServerCleanPathDir) Open(path string) (File, error) {
	*(d.log) = append(*(d.log), path)
	if path == "/" || path == "/dir" || path == "/dir/" {
		// Just return back something that's a directory.
		return Dir(".").Open(".")
	}
	return nil, fs.ErrNotExist
}

type panicOnSeek struct{ io.ReadSeeker }

func TestScanETag(t *testing.T) {
	tests := []struct {
		in         string
		wantETag   string
		wantRemain string
	}{
		{`W/"etag-1"`, `W/"etag-1"`, ""},
		{`"etag-2"`, `"etag-2"`, ""},
		{`"etag-1", "etag-2"`, `"etag-1"`, `, "etag-2"`},
		{"", "", ""},
		{"W/", "", ""},
		{`W/"truc`, "", ""},
		{`w/"case-sensitive"`, "", ""},
		{`"spaced etag"`, "", ""},
	}
	for _, test := range tests {
		etag, remain := ExportScanETag(test.in)
		if etag != test.wantETag || remain != test.wantRemain {
			t.Errorf("scanETag(%q)=%q %q, want %q %q", test.in, etag, remain, test.wantETag, test.wantRemain)
		}
	}
}

// Issue 40940: Ensure that we only accept non-negative suffix-lengths
// in "Range": "bytes=-N", and should reject "bytes=--2".
func TestServeFileRejectsInvalidSuffixLengths(t *testing.T) {
	run(t, testServeFileRejectsInvalidSuffixLengths, []testMode{http1Mode, https1Mode, http2Mode})
}
func testServeFileRejectsInvalidSuffixLengths(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, FileServer(Dir("testdata"))).ts

	tests := []struct {
		r        string
		wantCode int
		wantBody string
	}{
		{"bytes=--6", 416, "invalid range\n"},
		{"bytes=--0", 416, "invalid range\n"},
		{"bytes=---0", 416, "invalid range\n"},
		{"bytes=-6", 206, "hello\n"},
		{"bytes=6-", 206, "html says hello\n"},
		{"bytes=-6-", 416, "invalid range\n"},
		{"bytes=-0", 206, ""},
		{"bytes=", 200, "index.html says hello\n"},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.r, func(t *testing.T) {
			req, err := NewRequest("GET", cst.URL+"/index.html", nil)
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Range", tt.r)
			res, err := cst.Client().Do(req)
			if err != nil {
				t.Fatal(err)
			}
			if g, w := res.StatusCode, tt.wantCode; g != w {
				t.Errorf("StatusCode mismatch: got %d want %d", g, w)
			}
			slurp, err := io.ReadAll(res.Body)
			res.Body.Close()
			if err != nil {
				t.Fatal(err)
			}
			if g, w := string(slurp), tt.wantBody; g != w {
				t.Fatalf("Content mismatch:\nGot:  %q\nWant: %q", g, w)
			}
		})
	}
}

func TestFileServerMethods(t *testing.T) {
	run(t, testFileServerMethods)
}
func testFileServerMethods(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, FileServer(Dir("testdata"))).ts

	file, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatal("reading file:", err)
	}

	// Get contents via various methods.
	//
	// See https://go.dev/issue/59471 for a proposal to limit the set of methods handled.
	// For now, test the historical behavior.
	for _, method := range []string{
		MethodGet,
		MethodHead,
		MethodPost,
		MethodPut,
		MethodPatch,
		MethodDelete,
		MethodOptions,
		MethodTrace,
	} {
		req, _ := NewRequest(method, ts.URL+"/file", nil)
		t.Log(req.URL)
		res, err := ts.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		body, err := io.ReadAll(res.Body)
		res.Body.Close()
		if err != nil {
			t.Fatal(err)
		}
		wantBody := file
		if method == MethodHead {
			wantBody = nil
		}
		if !bytes.Equal(body, wantBody) {
			t.Fatalf("%v: got body %q, want %q", method, body, wantBody)
		}
		if got, want := res.Header.Get("Content-Length"), fmt.Sprint(len(file)); got != want {
			t.Fatalf("%v: got Content-Length %q, want %q", method, got, want)
		}
	}
}

func TestFileServerFS(t *testing.T) {
	filename := "index.html"
	contents := []byte("index.html says hello")
	fsys := fstest.MapFS{
		filename: {Data: contents},
	}
	ts := newClientServerTest(t, http1Mode, FileServerFS(fsys)).ts
	defer ts.Close()

	res, err := ts.Client().Get(ts.URL + "/" + filename)
	if err != nil {
		t.Fatal(err)
	}
	b, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal("reading Body:", err)
	}
	if s := string(b); s != string(contents) {
		t.Errorf("for path %q got %q, want %q", filename, s, contents)
	}
	res.Body.Close()
}

func TestServeFileFS(t *testing.T) {
	filename := "index.html"
	contents := []byte("index.html says hello")
	fsys := fstest.MapFS{
		filename: {Data: contents},
	}
	ts := newClientServerTest(t, http1Mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		ServeFileFS(w, r, fsys, filename)
	})).ts
	defer ts.Close()

	res, err := ts.Client().Get(ts.URL + "/" + filename)
	if err != nil {
		t.Fatal(err)
	}
	b, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal("reading Body:", err)
	}
	if s := string(b); s != string(contents) {
		t.Errorf("for path %q got %q, want %q", filename, s, contents)
	}
	res.Body.Close()
}

func TestServeFileZippingResponseWriter(t *testing.T) {
	// This test exercises a pattern which is incorrect,
	// but has been observed enough in the world that we don't want to break it.
	//
	// The server is setting "Content-Encoding: gzip",
	// wrapping the ResponseWriter in an implementation which gzips data written to it,
	// and passing this ResponseWriter to ServeFile.
	//
	// This means ServeFile cannot properly set a Content-Length header, because it
	// doesn't know what content it is going to send--the ResponseWriter is modifying
	// the bytes sent.
	//
	// Range requests are always going to be broken in this scenario,
	// but verify that we can serve non-range requests correctly.
	filename := "index.html"
	contents := []byte("contents will be sent with Content-Encoding: gzip")
	fsys := fstest.MapFS{
		filename: {Data: contents},
	}
	ts := newClientServerTest(t, http1Mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Encoding", "gzip")
		gzw := gzip.NewWriter(w)
		defer gzw.Close()
		ServeFileFS(gzipResponseWriter{w: gzw, ResponseWriter: w}, r, fsys, filename)
	})).ts
	defer ts.Close()

	res, err := ts.Client().Get(ts.URL + "/" + filename)
	if err != nil {
		t.Fatal(err)
	}
	b, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal("reading Body:", err)
	}
	if s := string(b); s != string(contents) {
		t.Errorf("for path %q got %q, want %q", filename, s, contents)
	}
	res.Body.Close()
}

type gzipResponseWriter struct {
	ResponseWriter
	w *gzip.Writer
}

func (grw gzipResponseWriter) Write(b []byte) (int, error) {
	return grw.w.Write(b)
}

func (grw gzipResponseWriter) Flush() {
	grw.w.Flush()
	if fw, ok := grw.ResponseWriter.(http.Flusher); ok {
		fw.Flush()
	}
}

// Issue 63769
func TestFileServerDirWithRootFile(t *testing.T) { run(t, testFileServerDirWithRootFile) }
func testFileServerDirWithRootFile(t *testing.T, mode testMode) {
	testDirFile := func(t *testing.T, h Handler) {
		ts := newClientServerTest(t, mode, h).ts
		defer ts.Close()

		res, err := ts.Client().Get(ts.URL)
		if err != nil {
			t.Fatal(err)
		}
		if g, w := res.StatusCode, StatusInternalServerError; g != w {
			t.Errorf("StatusCode mismatch: got %d, want: %d", g, w)
		}
		res.Body.Close()
	}

	t.Run("FileServer", func(t *testing.T) {
		testDirFile(t, FileServer(Dir("testdata/index.html")))
	})

	t.Run("FileServerFS", func(t *testing.T) {
		testDirFile(t, FileServerFS(os.DirFS("testdata/index.html")))
	})
}

func TestServeContentHeadersWithError(t *testing.T) {
	contents := []byte("content")
	ts := newClientServerTest(t, http1Mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("Content-Length", strconv.Itoa(len(contents)))
		w.Header().Set("Content-Encoding", "gzip")
		w.Header().Set("Etag", `"abcdefgh"`)
		w.Header().Set("Last-Modified", "Wed, 21 Oct 2015 07:28:00 GMT")
		w.Header().Set("Cache-Control", "immutable")
		w.Header().Set("Other-Header", "test")
		ServeContent(w, r, "", time.Time{}, bytes.NewReader(contents))
	})).ts
	defer ts.Close()

	req, err := NewRequest("GET", ts.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Range", "bytes=100-10000")

	c := ts.Client()
	res, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}

	out, _ := io.ReadAll(res.Body)
	res.Body.Close()

	if g, e := res.StatusCode, 416; g != e {
		t.Errorf("got status = %d; want %d", g, e)
	}
	if g, e := string(out), "invalid range: failed to overlap\n"; g != e {
		t.Errorf("got body = %q; want %q", g, e)
	}
	if g, e := res.Header.Get("Content-Type"), "text/plain; charset=utf-8"; g != e {
		t.Errorf("got content-type = %q, want %q", g, e)
	}
	if g, e := res.Header.Get("Content-Length"), strconv.Itoa(len(out)); g != e {
		t.Errorf("got content-length = %q, want %q", g, e)
	}
	if g, e := res.Header.Get("Content-Encoding"), ""; g != e {
		t.Errorf("got content-encoding = %q, want %q", g, e)
	}
	if g, e := res.Header.Get("Etag"), ""; g != e {
		t.Errorf("got etag = %q, want %q", g, e)
	}
	if g, e := res.Header.Get("Last-Modified"), ""; g != e {
		t.Errorf("got last-modified = %q, want %q", g, e)
	}
	if g, e := res.Header.Get("Cache-Control"), ""; g != e {
		t.Errorf("got cache-control = %q, want %q", g, e)
	}
	if g, e := res.Header.Get("Content-Range"), "bytes */7"; g != e {
		t.Errorf("got content-range = %q, want %q", g, e)
	}
	if g, e := res.Header.Get("Other-Header"), "test"; g != e {
		t.Errorf("got other-header = %q, want %q", g, e)
	}
}
