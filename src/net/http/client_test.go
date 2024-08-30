// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for client.go

package http_test

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/base64"
	"errors"
	"fmt"
	"internal/testenv"
	"io"
	"log"
	"net"
	. "net/http"
	"net/http/cookiejar"
	"net/http/httptest"
	"net/url"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

var robotsTxtHandler = HandlerFunc(func(w ResponseWriter, r *Request) {
	w.Header().Set("Last-Modified", "sometime")
	fmt.Fprintf(w, "User-agent: go\nDisallow: /something/")
})

// pedanticReadAll works like io.ReadAll but additionally
// verifies that r obeys the documented io.Reader contract.
func pedanticReadAll(r io.Reader) (b []byte, err error) {
	var bufa [64]byte
	buf := bufa[:]
	for {
		n, err := r.Read(buf)
		if n == 0 && err == nil {
			return nil, fmt.Errorf("Read: n=0 with err=nil")
		}
		b = append(b, buf[:n]...)
		if err == io.EOF {
			n, err := r.Read(buf)
			if n != 0 || err != io.EOF {
				return nil, fmt.Errorf("Read: n=%d err=%#v after EOF", n, err)
			}
			return b, nil
		}
		if err != nil {
			return b, err
		}
	}
}

func TestClient(t *testing.T) { run(t, testClient) }
func testClient(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, robotsTxtHandler).ts

	c := ts.Client()
	r, err := c.Get(ts.URL)
	var b []byte
	if err == nil {
		b, err = pedanticReadAll(r.Body)
		r.Body.Close()
	}
	if err != nil {
		t.Error(err)
	} else if s := string(b); !strings.HasPrefix(s, "User-agent:") {
		t.Errorf("Incorrect page body (did not begin with User-agent): %q", s)
	}
}

func TestClientHead(t *testing.T) { run(t, testClientHead) }
func testClientHead(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, robotsTxtHandler)
	r, err := cst.c.Head(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := r.Header["Last-Modified"]; !ok {
		t.Error("Last-Modified header not found.")
	}
}

type recordingTransport struct {
	req *Request
}

func (t *recordingTransport) RoundTrip(req *Request) (resp *Response, err error) {
	t.req = req
	return nil, errors.New("dummy impl")
}

func TestGetRequestFormat(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	tr := &recordingTransport{}
	client := &Client{Transport: tr}
	url := "http://dummy.faketld/"
	client.Get(url) // Note: doesn't hit network
	if tr.req.Method != "GET" {
		t.Errorf("expected method %q; got %q", "GET", tr.req.Method)
	}
	if tr.req.URL.String() != url {
		t.Errorf("expected URL %q; got %q", url, tr.req.URL.String())
	}
	if tr.req.Header == nil {
		t.Errorf("expected non-nil request Header")
	}
}

func TestPostRequestFormat(t *testing.T) {
	defer afterTest(t)
	tr := &recordingTransport{}
	client := &Client{Transport: tr}

	url := "http://dummy.faketld/"
	json := `{"key":"value"}`
	b := strings.NewReader(json)
	client.Post(url, "application/json", b) // Note: doesn't hit network

	if tr.req.Method != "POST" {
		t.Errorf("got method %q, want %q", tr.req.Method, "POST")
	}
	if tr.req.URL.String() != url {
		t.Errorf("got URL %q, want %q", tr.req.URL.String(), url)
	}
	if tr.req.Header == nil {
		t.Fatalf("expected non-nil request Header")
	}
	if tr.req.Close {
		t.Error("got Close true, want false")
	}
	if g, e := tr.req.ContentLength, int64(len(json)); g != e {
		t.Errorf("got ContentLength %d, want %d", g, e)
	}
}

func TestPostFormRequestFormat(t *testing.T) {
	defer afterTest(t)
	tr := &recordingTransport{}
	client := &Client{Transport: tr}

	urlStr := "http://dummy.faketld/"
	form := make(url.Values)
	form.Set("foo", "bar")
	form.Add("foo", "bar2")
	form.Set("bar", "baz")
	client.PostForm(urlStr, form) // Note: doesn't hit network

	if tr.req.Method != "POST" {
		t.Errorf("got method %q, want %q", tr.req.Method, "POST")
	}
	if tr.req.URL.String() != urlStr {
		t.Errorf("got URL %q, want %q", tr.req.URL.String(), urlStr)
	}
	if tr.req.Header == nil {
		t.Fatalf("expected non-nil request Header")
	}
	if g, e := tr.req.Header.Get("Content-Type"), "application/x-www-form-urlencoded"; g != e {
		t.Errorf("got Content-Type %q, want %q", g, e)
	}
	if tr.req.Close {
		t.Error("got Close true, want false")
	}
	// Depending on map iteration, body can be either of these.
	expectedBody := "foo=bar&foo=bar2&bar=baz"
	expectedBody1 := "bar=baz&foo=bar&foo=bar2"
	if g, e := tr.req.ContentLength, int64(len(expectedBody)); g != e {
		t.Errorf("got ContentLength %d, want %d", g, e)
	}
	bodyb, err := io.ReadAll(tr.req.Body)
	if err != nil {
		t.Fatalf("ReadAll on req.Body: %v", err)
	}
	if g := string(bodyb); g != expectedBody && g != expectedBody1 {
		t.Errorf("got body %q, want %q or %q", g, expectedBody, expectedBody1)
	}
}

func TestClientRedirects(t *testing.T) { run(t, testClientRedirects) }
func testClientRedirects(t *testing.T, mode testMode) {
	var ts *httptest.Server
	ts = newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		n, _ := strconv.Atoi(r.FormValue("n"))
		// Test Referer header. (7 is arbitrary position to test at)
		if n == 7 {
			if g, e := r.Referer(), ts.URL+"/?n=6"; e != g {
				t.Errorf("on request ?n=7, expected referer of %q; got %q", e, g)
			}
		}
		if n < 15 {
			Redirect(w, r, fmt.Sprintf("/?n=%d", n+1), StatusTemporaryRedirect)
			return
		}
		fmt.Fprintf(w, "n=%d", n)
	})).ts

	c := ts.Client()
	_, err := c.Get(ts.URL)
	if e, g := `Get "/?n=10": stopped after 10 redirects`, fmt.Sprintf("%v", err); e != g {
		t.Errorf("with default client Get, expected error %q, got %q", e, g)
	}

	// HEAD request should also have the ability to follow redirects.
	_, err = c.Head(ts.URL)
	if e, g := `Head "/?n=10": stopped after 10 redirects`, fmt.Sprintf("%v", err); e != g {
		t.Errorf("with default client Head, expected error %q, got %q", e, g)
	}

	// Do should also follow redirects.
	greq, _ := NewRequest("GET", ts.URL, nil)
	_, err = c.Do(greq)
	if e, g := `Get "/?n=10": stopped after 10 redirects`, fmt.Sprintf("%v", err); e != g {
		t.Errorf("with default client Do, expected error %q, got %q", e, g)
	}

	// Requests with an empty Method should also redirect (Issue 12705)
	greq.Method = ""
	_, err = c.Do(greq)
	if e, g := `Get "/?n=10": stopped after 10 redirects`, fmt.Sprintf("%v", err); e != g {
		t.Errorf("with default client Do and empty Method, expected error %q, got %q", e, g)
	}

	var checkErr error
	var lastVia []*Request
	var lastReq *Request
	c.CheckRedirect = func(req *Request, via []*Request) error {
		lastReq = req
		lastVia = via
		return checkErr
	}
	res, err := c.Get(ts.URL)
	if err != nil {
		t.Fatalf("Get error: %v", err)
	}
	res.Body.Close()
	finalURL := res.Request.URL.String()
	if e, g := "<nil>", fmt.Sprintf("%v", err); e != g {
		t.Errorf("with custom client, expected error %q, got %q", e, g)
	}
	if !strings.HasSuffix(finalURL, "/?n=15") {
		t.Errorf("expected final url to end in /?n=15; got url %q", finalURL)
	}
	if e, g := 15, len(lastVia); e != g {
		t.Errorf("expected lastVia to have contained %d elements; got %d", e, g)
	}

	// Test that Request.Cancel is propagated between requests (Issue 14053)
	creq, _ := NewRequest("HEAD", ts.URL, nil)
	cancel := make(chan struct{})
	creq.Cancel = cancel
	if _, err := c.Do(creq); err != nil {
		t.Fatal(err)
	}
	if lastReq == nil {
		t.Fatal("didn't see redirect")
	}
	if lastReq.Cancel != cancel {
		t.Errorf("expected lastReq to have the cancel channel set on the initial req")
	}

	checkErr = errors.New("no redirects allowed")
	res, err = c.Get(ts.URL)
	if urlError, ok := err.(*url.Error); !ok || urlError.Err != checkErr {
		t.Errorf("with redirects forbidden, expected a *url.Error with our 'no redirects allowed' error inside; got %#v (%q)", err, err)
	}
	if res == nil {
		t.Fatalf("Expected a non-nil Response on CheckRedirect failure (https://golang.org/issue/3795)")
	}
	res.Body.Close()
	if res.Header.Get("Location") == "" {
		t.Errorf("no Location header in Response")
	}
}

// Tests that Client redirects' contexts are derived from the original request's context.
func TestClientRedirectsContext(t *testing.T) { run(t, testClientRedirectsContext) }
func testClientRedirectsContext(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		Redirect(w, r, "/", StatusTemporaryRedirect)
	})).ts

	ctx, cancel := context.WithCancel(context.Background())
	c := ts.Client()
	c.CheckRedirect = func(req *Request, via []*Request) error {
		cancel()
		select {
		case <-req.Context().Done():
			return nil
		case <-time.After(5 * time.Second):
			return errors.New("redirected request's context never expired after root request canceled")
		}
	}
	req, _ := NewRequestWithContext(ctx, "GET", ts.URL, nil)
	_, err := c.Do(req)
	ue, ok := err.(*url.Error)
	if !ok {
		t.Fatalf("got error %T; want *url.Error", err)
	}
	if ue.Err != context.Canceled {
		t.Errorf("url.Error.Err = %v; want %v", ue.Err, context.Canceled)
	}
}

type redirectTest struct {
	suffix       string
	want         int // response code
	redirectBody string
}

func TestPostRedirects(t *testing.T) {
	postRedirectTests := []redirectTest{
		{"/", 200, "first"},
		{"/?code=301&next=302", 200, "c301"},
		{"/?code=302&next=302", 200, "c302"},
		{"/?code=303&next=301", 200, "c303wc301"}, // Issue 9348
		{"/?code=304", 304, "c304"},
		{"/?code=305", 305, "c305"},
		{"/?code=307&next=303,308,302", 200, "c307"},
		{"/?code=308&next=302,301", 200, "c308"},
		{"/?code=404", 404, "c404"},
	}

	wantSegments := []string{
		`POST / "first"`,
		`POST /?code=301&next=302 "c301"`,
		`GET /?code=302 ""`,
		`GET / ""`,
		`POST /?code=302&next=302 "c302"`,
		`GET /?code=302 ""`,
		`GET / ""`,
		`POST /?code=303&next=301 "c303wc301"`,
		`GET /?code=301 ""`,
		`GET / ""`,
		`POST /?code=304 "c304"`,
		`POST /?code=305 "c305"`,
		`POST /?code=307&next=303,308,302 "c307"`,
		`POST /?code=303&next=308,302 "c307"`,
		`GET /?code=308&next=302 ""`,
		`GET /?code=302 "c307"`,
		`GET / ""`,
		`POST /?code=308&next=302,301 "c308"`,
		`POST /?code=302&next=301 "c308"`,
		`GET /?code=301 ""`,
		`GET / ""`,
		`POST /?code=404 "c404"`,
	}
	want := strings.Join(wantSegments, "\n")
	run(t, func(t *testing.T, mode testMode) {
		testRedirectsByMethod(t, mode, "POST", postRedirectTests, want)
	})
}

func TestDeleteRedirects(t *testing.T) {
	deleteRedirectTests := []redirectTest{
		{"/", 200, "first"},
		{"/?code=301&next=302,308", 200, "c301"},
		{"/?code=302&next=302", 200, "c302"},
		{"/?code=303", 200, "c303"},
		{"/?code=307&next=301,308,303,302,304", 304, "c307"},
		{"/?code=308&next=307", 200, "c308"},
		{"/?code=404", 404, "c404"},
	}

	wantSegments := []string{
		`DELETE / "first"`,
		`DELETE /?code=301&next=302,308 "c301"`,
		`GET /?code=302&next=308 ""`,
		`GET /?code=308 ""`,
		`GET / "c301"`,
		`DELETE /?code=302&next=302 "c302"`,
		`GET /?code=302 ""`,
		`GET / ""`,
		`DELETE /?code=303 "c303"`,
		`GET / ""`,
		`DELETE /?code=307&next=301,308,303,302,304 "c307"`,
		`DELETE /?code=301&next=308,303,302,304 "c307"`,
		`GET /?code=308&next=303,302,304 ""`,
		`GET /?code=303&next=302,304 "c307"`,
		`GET /?code=302&next=304 ""`,
		`GET /?code=304 ""`,
		`DELETE /?code=308&next=307 "c308"`,
		`DELETE /?code=307 "c308"`,
		`DELETE / "c308"`,
		`DELETE /?code=404 "c404"`,
	}
	want := strings.Join(wantSegments, "\n")
	run(t, func(t *testing.T, mode testMode) {
		testRedirectsByMethod(t, mode, "DELETE", deleteRedirectTests, want)
	})
}

func testRedirectsByMethod(t *testing.T, mode testMode, method string, table []redirectTest, want string) {
	var log struct {
		sync.Mutex
		bytes.Buffer
	}
	var ts *httptest.Server
	ts = newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		log.Lock()
		slurp, _ := io.ReadAll(r.Body)
		fmt.Fprintf(&log.Buffer, "%s %s %q", r.Method, r.RequestURI, slurp)
		if cl := r.Header.Get("Content-Length"); r.Method == "GET" && len(slurp) == 0 && (r.ContentLength != 0 || cl != "") {
			fmt.Fprintf(&log.Buffer, " (but with body=%T, content-length = %v, %q)", r.Body, r.ContentLength, cl)
		}
		log.WriteByte('\n')
		log.Unlock()
		urlQuery := r.URL.Query()
		if v := urlQuery.Get("code"); v != "" {
			location := ts.URL
			if final := urlQuery.Get("next"); final != "" {
				first, rest, _ := strings.Cut(final, ",")
				location = fmt.Sprintf("%s?code=%s", location, first)
				if rest != "" {
					location = fmt.Sprintf("%s&next=%s", location, rest)
				}
			}
			code, _ := strconv.Atoi(v)
			if code/100 == 3 {
				w.Header().Set("Location", location)
			}
			w.WriteHeader(code)
		}
	})).ts

	c := ts.Client()
	for _, tt := range table {
		content := tt.redirectBody
		req, _ := NewRequest(method, ts.URL+tt.suffix, strings.NewReader(content))
		req.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(strings.NewReader(content)), nil }
		res, err := c.Do(req)

		if err != nil {
			t.Fatal(err)
		}
		if res.StatusCode != tt.want {
			t.Errorf("POST %s: status code = %d; want %d", tt.suffix, res.StatusCode, tt.want)
		}
	}
	log.Lock()
	got := log.String()
	log.Unlock()

	got = strings.TrimSpace(got)
	want = strings.TrimSpace(want)

	if got != want {
		got, want, lines := removeCommonLines(got, want)
		t.Errorf("Log differs after %d common lines.\n\nGot:\n%s\n\nWant:\n%s\n", lines, got, want)
	}
}

func removeCommonLines(a, b string) (asuffix, bsuffix string, commonLines int) {
	for {
		nl := strings.IndexByte(a, '\n')
		if nl < 0 {
			return a, b, commonLines
		}
		line := a[:nl+1]
		if !strings.HasPrefix(b, line) {
			return a, b, commonLines
		}
		commonLines++
		a = a[len(line):]
		b = b[len(line):]
	}
}

func TestClientRedirectUseResponse(t *testing.T) { run(t, testClientRedirectUseResponse) }
func testClientRedirectUseResponse(t *testing.T, mode testMode) {
	const body = "Hello, world."
	var ts *httptest.Server
	ts = newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		if strings.Contains(r.URL.Path, "/other") {
			io.WriteString(w, "wrong body")
		} else {
			w.Header().Set("Location", ts.URL+"/other")
			w.WriteHeader(StatusFound)
			io.WriteString(w, body)
		}
	})).ts

	c := ts.Client()
	c.CheckRedirect = func(req *Request, via []*Request) error {
		if req.Response == nil {
			t.Error("expected non-nil Request.Response")
		}
		return ErrUseLastResponse
	}
	res, err := c.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != StatusFound {
		t.Errorf("status = %d; want %d", res.StatusCode, StatusFound)
	}
	defer res.Body.Close()
	slurp, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(slurp) != body {
		t.Errorf("body = %q; want %q", slurp, body)
	}
}

// Issues 17773 and 49281: don't follow a 3xx if the response doesn't
// have a Location header.
func TestClientRedirectNoLocation(t *testing.T) { run(t, testClientRedirectNoLocation) }
func testClientRedirectNoLocation(t *testing.T, mode testMode) {
	for _, code := range []int{301, 308} {
		t.Run(fmt.Sprint(code), func(t *testing.T) {
			setParallel(t)
			cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
				w.Header().Set("Foo", "Bar")
				w.WriteHeader(code)
			}))
			res, err := cst.c.Get(cst.ts.URL)
			if err != nil {
				t.Fatal(err)
			}
			res.Body.Close()
			if res.StatusCode != code {
				t.Errorf("status = %d; want %d", res.StatusCode, code)
			}
			if got := res.Header.Get("Foo"); got != "Bar" {
				t.Errorf("Foo header = %q; want Bar", got)
			}
		})
	}
}

// Don't follow a 307/308 if we can't resent the request body.
func TestClientRedirect308NoGetBody(t *testing.T) { run(t, testClientRedirect308NoGetBody) }
func testClientRedirect308NoGetBody(t *testing.T, mode testMode) {
	const fakeURL = "https://localhost:1234/" // won't be hit
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Location", fakeURL)
		w.WriteHeader(308)
	})).ts
	req, err := NewRequest("POST", ts.URL, strings.NewReader("some body"))
	if err != nil {
		t.Fatal(err)
	}
	c := ts.Client()
	req.GetBody = nil // so it can't rewind.
	res, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.StatusCode != 308 {
		t.Errorf("status = %d; want %d", res.StatusCode, 308)
	}
	if got := res.Header.Get("Location"); got != fakeURL {
		t.Errorf("Location header = %q; want %q", got, fakeURL)
	}
}

var expectedCookies = []*Cookie{
	{Name: "ChocolateChip", Value: "tasty"},
	{Name: "First", Value: "Hit"},
	{Name: "Second", Value: "Hit"},
}

var echoCookiesRedirectHandler = HandlerFunc(func(w ResponseWriter, r *Request) {
	for _, cookie := range r.Cookies() {
		SetCookie(w, cookie)
	}
	if r.URL.Path == "/" {
		SetCookie(w, expectedCookies[1])
		Redirect(w, r, "/second", StatusMovedPermanently)
	} else {
		SetCookie(w, expectedCookies[2])
		w.Write([]byte("hello"))
	}
})

func TestClientSendsCookieFromJar(t *testing.T) {
	defer afterTest(t)
	tr := &recordingTransport{}
	client := &Client{Transport: tr}
	client.Jar = &TestJar{perURL: make(map[string][]*Cookie)}
	us := "http://dummy.faketld/"
	u, _ := url.Parse(us)
	client.Jar.SetCookies(u, expectedCookies)

	client.Get(us) // Note: doesn't hit network
	matchReturnedCookies(t, expectedCookies, tr.req.Cookies())

	client.Head(us) // Note: doesn't hit network
	matchReturnedCookies(t, expectedCookies, tr.req.Cookies())

	client.Post(us, "text/plain", strings.NewReader("body")) // Note: doesn't hit network
	matchReturnedCookies(t, expectedCookies, tr.req.Cookies())

	client.PostForm(us, url.Values{}) // Note: doesn't hit network
	matchReturnedCookies(t, expectedCookies, tr.req.Cookies())

	req, _ := NewRequest("GET", us, nil)
	client.Do(req) // Note: doesn't hit network
	matchReturnedCookies(t, expectedCookies, tr.req.Cookies())

	req, _ = NewRequest("POST", us, nil)
	client.Do(req) // Note: doesn't hit network
	matchReturnedCookies(t, expectedCookies, tr.req.Cookies())
}

// Just enough correctness for our redirect tests. Uses the URL.Host as the
// scope of all cookies.
type TestJar struct {
	m      sync.Mutex
	perURL map[string][]*Cookie
}

func (j *TestJar) SetCookies(u *url.URL, cookies []*Cookie) {
	j.m.Lock()
	defer j.m.Unlock()
	if j.perURL == nil {
		j.perURL = make(map[string][]*Cookie)
	}
	j.perURL[u.Host] = cookies
}

func (j *TestJar) Cookies(u *url.URL) []*Cookie {
	j.m.Lock()
	defer j.m.Unlock()
	return j.perURL[u.Host]
}

func TestRedirectCookiesJar(t *testing.T) { run(t, testRedirectCookiesJar) }
func testRedirectCookiesJar(t *testing.T, mode testMode) {
	var ts *httptest.Server
	ts = newClientServerTest(t, mode, echoCookiesRedirectHandler).ts
	c := ts.Client()
	c.Jar = new(TestJar)
	u, _ := url.Parse(ts.URL)
	c.Jar.SetCookies(u, []*Cookie{expectedCookies[0]})
	resp, err := c.Get(ts.URL)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	resp.Body.Close()
	matchReturnedCookies(t, expectedCookies, resp.Cookies())
}

func matchReturnedCookies(t *testing.T, expected, given []*Cookie) {
	if len(given) != len(expected) {
		t.Logf("Received cookies: %v", given)
		t.Errorf("Expected %d cookies, got %d", len(expected), len(given))
	}
	for _, ec := range expected {
		foundC := false
		for _, c := range given {
			if ec.Name == c.Name && ec.Value == c.Value {
				foundC = true
				break
			}
		}
		if !foundC {
			t.Errorf("Missing cookie %v", ec)
		}
	}
}

func TestJarCalls(t *testing.T) { run(t, testJarCalls, []testMode{http1Mode}) }
func testJarCalls(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		pathSuffix := r.RequestURI[1:]
		if r.RequestURI == "/nosetcookie" {
			return // don't set cookies for this path
		}
		SetCookie(w, &Cookie{Name: "name" + pathSuffix, Value: "val" + pathSuffix})
		if r.RequestURI == "/" {
			Redirect(w, r, "http://secondhost.fake/secondpath", 302)
		}
	})).ts
	jar := new(RecordingJar)
	c := ts.Client()
	c.Jar = jar
	c.Transport.(*Transport).Dial = func(_ string, _ string) (net.Conn, error) {
		return net.Dial("tcp", ts.Listener.Addr().String())
	}
	_, err := c.Get("http://firsthost.fake/")
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Get("http://firsthost.fake/nosetcookie")
	if err != nil {
		t.Fatal(err)
	}
	got := jar.log.String()
	want := `Cookies("http://firsthost.fake/")
SetCookie("http://firsthost.fake/", [name=val])
Cookies("http://secondhost.fake/secondpath")
SetCookie("http://secondhost.fake/secondpath", [namesecondpath=valsecondpath])
Cookies("http://firsthost.fake/nosetcookie")
`
	if got != want {
		t.Errorf("Got Jar calls:\n%s\nWant:\n%s", got, want)
	}
}

// RecordingJar keeps a log of calls made to it, without
// tracking any cookies.
type RecordingJar struct {
	mu  sync.Mutex
	log bytes.Buffer
}

func (j *RecordingJar) SetCookies(u *url.URL, cookies []*Cookie) {
	j.logf("SetCookie(%q, %v)\n", u, cookies)
}

func (j *RecordingJar) Cookies(u *url.URL) []*Cookie {
	j.logf("Cookies(%q)\n", u)
	return nil
}

func (j *RecordingJar) logf(format string, args ...any) {
	j.mu.Lock()
	defer j.mu.Unlock()
	fmt.Fprintf(&j.log, format, args...)
}

func TestStreamingGet(t *testing.T) { run(t, testStreamingGet) }
func testStreamingGet(t *testing.T, mode testMode) {
	say := make(chan string)
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.(Flusher).Flush()
		for str := range say {
			w.Write([]byte(str))
			w.(Flusher).Flush()
		}
	}))

	c := cst.c
	res, err := c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	var buf [10]byte
	for _, str := range []string{"i", "am", "also", "known", "as", "comet"} {
		say <- str
		n, err := io.ReadFull(res.Body, buf[:len(str)])
		if err != nil {
			t.Fatalf("ReadFull on %q: %v", str, err)
		}
		if n != len(str) {
			t.Fatalf("Receiving %q, only read %d bytes", str, n)
		}
		got := string(buf[0:n])
		if got != str {
			t.Fatalf("Expected %q, got %q", str, got)
		}
	}
	close(say)
	_, err = io.ReadFull(res.Body, buf[0:1])
	if err != io.EOF {
		t.Fatalf("at end expected EOF, got %v", err)
	}
}

type writeCountingConn struct {
	net.Conn
	count *int
}

func (c *writeCountingConn) Write(p []byte) (int, error) {
	*c.count++
	return c.Conn.Write(p)
}

// TestClientWrites verifies that client requests are buffered and we
// don't send a TCP packet per line of the http request + body.
func TestClientWrites(t *testing.T) { run(t, testClientWrites, []testMode{http1Mode}) }
func testClientWrites(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
	})).ts

	writes := 0
	dialer := func(netz string, addr string) (net.Conn, error) {
		c, err := net.Dial(netz, addr)
		if err == nil {
			c = &writeCountingConn{c, &writes}
		}
		return c, err
	}
	c := ts.Client()
	c.Transport.(*Transport).Dial = dialer

	_, err := c.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if writes != 1 {
		t.Errorf("Get request did %d Write calls, want 1", writes)
	}

	writes = 0
	_, err = c.PostForm(ts.URL, url.Values{"foo": {"bar"}})
	if err != nil {
		t.Fatal(err)
	}
	if writes != 1 {
		t.Errorf("Post request did %d Write calls, want 1", writes)
	}
}

func TestClientInsecureTransport(t *testing.T) {
	run(t, testClientInsecureTransport, []testMode{https1Mode, http2Mode})
}
func testClientInsecureTransport(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Write([]byte("Hello"))
	}))
	ts := cst.ts
	errLog := new(strings.Builder)
	ts.Config.ErrorLog = log.New(errLog, "", 0)

	// TODO(bradfitz): add tests for skipping hostname checks too?
	// would require a new cert for testing, and probably
	// redundant with these tests.
	c := ts.Client()
	for _, insecure := range []bool{true, false} {
		c.Transport.(*Transport).TLSClientConfig = &tls.Config{
			InsecureSkipVerify: insecure,
		}
		res, err := c.Get(ts.URL)
		if (err == nil) != insecure {
			t.Errorf("insecure=%v: got unexpected err=%v", insecure, err)
		}
		if res != nil {
			res.Body.Close()
		}
	}

	cst.close()
	if !strings.Contains(errLog.String(), "TLS handshake error") {
		t.Errorf("expected an error log message containing 'TLS handshake error'; got %q", errLog)
	}
}

func TestClientErrorWithRequestURI(t *testing.T) {
	defer afterTest(t)
	req, _ := NewRequest("GET", "http://localhost:1234/", nil)
	req.RequestURI = "/this/field/is/illegal/and/should/error/"
	_, err := DefaultClient.Do(req)
	if err == nil {
		t.Fatalf("expected an error")
	}
	if !strings.Contains(err.Error(), "RequestURI") {
		t.Errorf("wanted error mentioning RequestURI; got error: %v", err)
	}
}

func TestClientWithCorrectTLSServerName(t *testing.T) {
	run(t, testClientWithCorrectTLSServerName, []testMode{https1Mode, http2Mode})
}
func testClientWithCorrectTLSServerName(t *testing.T, mode testMode) {
	const serverName = "example.com"
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		if r.TLS.ServerName != serverName {
			t.Errorf("expected client to set ServerName %q, got: %q", serverName, r.TLS.ServerName)
		}
	})).ts

	c := ts.Client()
	c.Transport.(*Transport).TLSClientConfig.ServerName = serverName
	if _, err := c.Get(ts.URL); err != nil {
		t.Fatalf("expected successful TLS connection, got error: %v", err)
	}
}

func TestClientWithIncorrectTLSServerName(t *testing.T) {
	run(t, testClientWithIncorrectTLSServerName, []testMode{https1Mode, http2Mode})
}
func testClientWithIncorrectTLSServerName(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {}))
	ts := cst.ts
	errLog := new(strings.Builder)
	ts.Config.ErrorLog = log.New(errLog, "", 0)

	c := ts.Client()
	c.Transport.(*Transport).TLSClientConfig.ServerName = "badserver"
	_, err := c.Get(ts.URL)
	if err == nil {
		t.Fatalf("expected an error")
	}
	if !strings.Contains(err.Error(), "127.0.0.1") || !strings.Contains(err.Error(), "badserver") {
		t.Errorf("wanted error mentioning 127.0.0.1 and badserver; got error: %v", err)
	}

	cst.close()
	if !strings.Contains(errLog.String(), "TLS handshake error") {
		t.Errorf("expected an error log message containing 'TLS handshake error'; got %q", errLog)
	}
}

// Test for golang.org/issue/5829; the Transport should respect TLSClientConfig.ServerName
// when not empty.
//
// tls.Config.ServerName (non-empty, set to "example.com") takes
// precedence over "some-other-host.tld" which previously incorrectly
// took precedence. We don't actually connect to (or even resolve)
// "some-other-host.tld", though, because of the Transport.Dial hook.
//
// The httptest.Server has a cert with "example.com" as its name.
func TestTransportUsesTLSConfigServerName(t *testing.T) {
	run(t, testTransportUsesTLSConfigServerName, []testMode{https1Mode, http2Mode})
}
func testTransportUsesTLSConfigServerName(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Write([]byte("Hello"))
	})).ts

	c := ts.Client()
	tr := c.Transport.(*Transport)
	tr.TLSClientConfig.ServerName = "example.com" // one of httptest's Server cert names
	tr.Dial = func(netw, addr string) (net.Conn, error) {
		return net.Dial(netw, ts.Listener.Addr().String())
	}
	res, err := c.Get("https://some-other-host.tld/")
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
}

func TestResponseSetsTLSConnectionState(t *testing.T) {
	run(t, testResponseSetsTLSConnectionState, []testMode{https1Mode})
}
func testResponseSetsTLSConnectionState(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Write([]byte("Hello"))
	})).ts

	c := ts.Client()
	tr := c.Transport.(*Transport)
	tr.TLSClientConfig.CipherSuites = []uint16{tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256}
	tr.TLSClientConfig.MaxVersion = tls.VersionTLS12 // to get to pick the cipher suite
	tr.Dial = func(netw, addr string) (net.Conn, error) {
		return net.Dial(netw, ts.Listener.Addr().String())
	}
	res, err := c.Get("https://example.com/")
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.TLS == nil {
		t.Fatal("Response didn't set TLS Connection State.")
	}
	if got, want := res.TLS.CipherSuite, tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256; got != want {
		t.Errorf("TLS Cipher Suite = %d; want %d", got, want)
	}
}

// Check that an HTTPS client can interpret a particular TLS error
// to determine that the server is speaking HTTP.
// See golang.org/issue/11111.
func TestHTTPSClientDetectsHTTPServer(t *testing.T) {
	run(t, testHTTPSClientDetectsHTTPServer, []testMode{http1Mode})
}
func testHTTPSClientDetectsHTTPServer(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {})).ts
	ts.Config.ErrorLog = quietLog

	_, err := Get(strings.Replace(ts.URL, "http", "https", 1))
	if got := err.Error(); !strings.Contains(got, "HTTP response to HTTPS client") {
		t.Fatalf("error = %q; want error indicating HTTP response to HTTPS request", got)
	}
}

// Verify Response.ContentLength is populated. https://golang.org/issue/4126
func TestClientHeadContentLength(t *testing.T) { run(t, testClientHeadContentLength) }
func testClientHeadContentLength(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		if v := r.FormValue("cl"); v != "" {
			w.Header().Set("Content-Length", v)
		}
	}))
	tests := []struct {
		suffix string
		want   int64
	}{
		{"/?cl=1234", 1234},
		{"/?cl=0", 0},
		{"", -1},
	}
	for _, tt := range tests {
		req, _ := NewRequest("HEAD", cst.ts.URL+tt.suffix, nil)
		res, err := cst.c.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		if res.ContentLength != tt.want {
			t.Errorf("Content-Length = %d; want %d", res.ContentLength, tt.want)
		}
		bs, err := io.ReadAll(res.Body)
		if err != nil {
			t.Fatal(err)
		}
		if len(bs) != 0 {
			t.Errorf("Unexpected content: %q", bs)
		}
	}
}

func TestEmptyPasswordAuth(t *testing.T) { run(t, testEmptyPasswordAuth) }
func testEmptyPasswordAuth(t *testing.T, mode testMode) {
	gopher := "gopher"
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		auth := r.Header.Get("Authorization")
		if strings.HasPrefix(auth, "Basic ") {
			encoded := auth[6:]
			decoded, err := base64.StdEncoding.DecodeString(encoded)
			if err != nil {
				t.Fatal(err)
			}
			expected := gopher + ":"
			s := string(decoded)
			if expected != s {
				t.Errorf("Invalid Authorization header. Got %q, wanted %q", s, expected)
			}
		} else {
			t.Errorf("Invalid auth %q", auth)
		}
	})).ts
	defer ts.Close()
	req, err := NewRequest("GET", ts.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.URL.User = url.User(gopher)
	c := ts.Client()
	resp, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
}

func TestBasicAuth(t *testing.T) {
	defer afterTest(t)
	tr := &recordingTransport{}
	client := &Client{Transport: tr}

	url := "http://My%20User:My%20Pass@dummy.faketld/"
	expected := "My User:My Pass"
	client.Get(url)

	if tr.req.Method != "GET" {
		t.Errorf("got method %q, want %q", tr.req.Method, "GET")
	}
	if tr.req.URL.String() != url {
		t.Errorf("got URL %q, want %q", tr.req.URL.String(), url)
	}
	if tr.req.Header == nil {
		t.Fatalf("expected non-nil request Header")
	}
	auth := tr.req.Header.Get("Authorization")
	if strings.HasPrefix(auth, "Basic ") {
		encoded := auth[6:]
		decoded, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			t.Fatal(err)
		}
		s := string(decoded)
		if expected != s {
			t.Errorf("Invalid Authorization header. Got %q, wanted %q", s, expected)
		}
	} else {
		t.Errorf("Invalid auth %q", auth)
	}
}

func TestBasicAuthHeadersPreserved(t *testing.T) {
	defer afterTest(t)
	tr := &recordingTransport{}
	client := &Client{Transport: tr}

	// If Authorization header is provided, username in URL should not override it
	url := "http://My%20User@dummy.faketld/"
	req, err := NewRequest("GET", url, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.SetBasicAuth("My User", "My Pass")
	expected := "My User:My Pass"
	client.Do(req)

	if tr.req.Method != "GET" {
		t.Errorf("got method %q, want %q", tr.req.Method, "GET")
	}
	if tr.req.URL.String() != url {
		t.Errorf("got URL %q, want %q", tr.req.URL.String(), url)
	}
	if tr.req.Header == nil {
		t.Fatalf("expected non-nil request Header")
	}
	auth := tr.req.Header.Get("Authorization")
	if strings.HasPrefix(auth, "Basic ") {
		encoded := auth[6:]
		decoded, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			t.Fatal(err)
		}
		s := string(decoded)
		if expected != s {
			t.Errorf("Invalid Authorization header. Got %q, wanted %q", s, expected)
		}
	} else {
		t.Errorf("Invalid auth %q", auth)
	}

}

func TestStripPasswordFromError(t *testing.T) {
	client := &Client{Transport: &recordingTransport{}}
	testCases := []struct {
		desc string
		in   string
		out  string
	}{
		{
			desc: "Strip password from error message",
			in:   "http://user:password@dummy.faketld/",
			out:  `Get "http://user:***@dummy.faketld/": dummy impl`,
		},
		{
			desc: "Don't Strip password from domain name",
			in:   "http://user:password@password.faketld/",
			out:  `Get "http://user:***@password.faketld/": dummy impl`,
		},
		{
			desc: "Don't Strip password from path",
			in:   "http://user:password@dummy.faketld/password",
			out:  `Get "http://user:***@dummy.faketld/password": dummy impl`,
		},
		{
			desc: "Strip escaped password",
			in:   "http://user:pa%2Fssword@dummy.faketld/",
			out:  `Get "http://user:***@dummy.faketld/": dummy impl`,
		},
	}
	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			_, err := client.Get(tC.in)
			if err.Error() != tC.out {
				t.Errorf("Unexpected output for %q: expected %q, actual %q",
					tC.in, tC.out, err.Error())
			}
		})
	}
}

func TestClientTimeout(t *testing.T) { run(t, testClientTimeout) }
func testClientTimeout(t *testing.T, mode testMode) {
	var (
		mu           sync.Mutex
		nonce        string // a unique per-request string
		sawSlowNonce bool   // true if the handler saw /slow?nonce=<nonce>
	)
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		_ = r.ParseForm()
		if r.URL.Path == "/" {
			Redirect(w, r, "/slow?nonce="+r.Form.Get("nonce"), StatusFound)
			return
		}
		if r.URL.Path == "/slow" {
			mu.Lock()
			if r.Form.Get("nonce") == nonce {
				sawSlowNonce = true
			} else {
				t.Logf("mismatched nonce: received %s, want %s", r.Form.Get("nonce"), nonce)
			}
			mu.Unlock()

			w.Write([]byte("Hello"))
			w.(Flusher).Flush()
			<-r.Context().Done()
			return
		}
	}))

	// Try to trigger a timeout after reading part of the response body.
	// The initial timeout is empirically usually long enough on a decently fast
	// machine, but if we undershoot we'll retry with exponentially longer
	// timeouts until the test either passes or times out completely.
	// This keeps the test reasonably fast in the typical case but allows it to
	// also eventually succeed on arbitrarily slow machines.
	timeout := 10 * time.Millisecond
	nextNonce := 0
	for ; ; timeout *= 2 {
		if timeout <= 0 {
			// The only way we can feasibly hit this while the test is running is if
			// the request fails without actually waiting for the timeout to occur.
			t.Fatalf("timeout overflow")
		}
		if deadline, ok := t.Deadline(); ok && !time.Now().Add(timeout).Before(deadline) {
			t.Fatalf("failed to produce expected timeout before test deadline")
		}
		t.Logf("attempting test with timeout %v", timeout)
		cst.c.Timeout = timeout

		mu.Lock()
		nonce = fmt.Sprint(nextNonce)
		nextNonce++
		sawSlowNonce = false
		mu.Unlock()
		res, err := cst.c.Get(cst.ts.URL + "/?nonce=" + nonce)
		if err != nil {
			if strings.Contains(err.Error(), "Client.Timeout") {
				// Timed out before handler could respond.
				t.Logf("timeout before response received")
				continue
			}
			if runtime.GOOS == "windows" && strings.HasPrefix(runtime.GOARCH, "arm") {
				testenv.SkipFlaky(t, 43120)
			}
			t.Fatal(err)
		}

		mu.Lock()
		ok := sawSlowNonce
		mu.Unlock()
		if !ok {
			t.Fatal("handler never got /slow request, but client returned response")
		}

		_, err = io.ReadAll(res.Body)
		res.Body.Close()

		if err == nil {
			t.Fatal("expected error from ReadAll")
		}
		ne, ok := err.(net.Error)
		if !ok {
			t.Errorf("error value from ReadAll was %T; expected some net.Error", err)
		} else if !ne.Timeout() {
			t.Errorf("net.Error.Timeout = false; want true")
		}
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("ReadAll error = %q; expected some context.DeadlineExceeded", err)
		}
		if got := ne.Error(); !strings.Contains(got, "(Client.Timeout") {
			if runtime.GOOS == "windows" && strings.HasPrefix(runtime.GOARCH, "arm") {
				testenv.SkipFlaky(t, 43120)
			}
			t.Errorf("error string = %q; missing timeout substring", got)
		}

		break
	}
}

// Client.Timeout firing before getting to the body
func TestClientTimeout_Headers(t *testing.T) { run(t, testClientTimeout_Headers) }
func testClientTimeout_Headers(t *testing.T, mode testMode) {
	donec := make(chan bool, 1)
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		<-donec
	}), optQuietLog)
	// Note that we use a channel send here and not a close.
	// The race detector doesn't know that we're waiting for a timeout
	// and thinks that the waitgroup inside httptest.Server is added to concurrently
	// with us closing it. If we timed out immediately, we could close the testserver
	// before we entered the handler. We're not timing out immediately and there's
	// no way we would be done before we entered the handler, but the race detector
	// doesn't know this, so synchronize explicitly.
	defer func() { donec <- true }()

	cst.c.Timeout = 5 * time.Millisecond
	res, err := cst.c.Get(cst.ts.URL)
	if err == nil {
		res.Body.Close()
		t.Fatal("got response from Get; expected error")
	}
	if _, ok := err.(*url.Error); !ok {
		t.Fatalf("Got error of type %T; want *url.Error", err)
	}
	ne, ok := err.(net.Error)
	if !ok {
		t.Fatalf("Got error of type %T; want some net.Error", err)
	}
	if !ne.Timeout() {
		t.Error("net.Error.Timeout = false; want true")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("ReadAll error = %q; expected some context.DeadlineExceeded", err)
	}
	if got := ne.Error(); !strings.Contains(got, "Client.Timeout exceeded") {
		if runtime.GOOS == "windows" && strings.HasPrefix(runtime.GOARCH, "arm") {
			testenv.SkipFlaky(t, 43120)
		}
		t.Errorf("error string = %q; missing timeout substring", got)
	}
}

// Issue 16094: if Client.Timeout is set but not hit, a Timeout error shouldn't be
// returned.
func TestClientTimeoutCancel(t *testing.T) { run(t, testClientTimeoutCancel) }
func testClientTimeoutCancel(t *testing.T, mode testMode) {
	testDone := make(chan struct{})
	ctx, cancel := context.WithCancel(context.Background())

	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.(Flusher).Flush()
		<-testDone
	}))
	defer close(testDone)

	cst.c.Timeout = 1 * time.Hour
	req, _ := NewRequest("GET", cst.ts.URL, nil)
	req.Cancel = ctx.Done()
	res, err := cst.c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	_, err = io.Copy(io.Discard, res.Body)
	if err != ExportErrRequestCanceled {
		t.Fatalf("error = %v; want errRequestCanceled", err)
	}
}

// Issue 49366: if Client.Timeout is set but not hit, no error should be returned.
func TestClientTimeoutDoesNotExpire(t *testing.T) { run(t, testClientTimeoutDoesNotExpire) }
func testClientTimeoutDoesNotExpire(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Write([]byte("body"))
	}))

	cst.c.Timeout = 1 * time.Hour
	req, _ := NewRequest("GET", cst.ts.URL, nil)
	res, err := cst.c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = io.Copy(io.Discard, res.Body); err != nil {
		t.Fatalf("io.Copy(io.Discard, res.Body) = %v, want nil", err)
	}
	if err = res.Body.Close(); err != nil {
		t.Fatalf("res.Body.Close() = %v, want nil", err)
	}
}

func TestClientRedirectEatsBody_h1(t *testing.T) { run(t, testClientRedirectEatsBody) }
func testClientRedirectEatsBody(t *testing.T, mode testMode) {
	saw := make(chan string, 2)
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		saw <- r.RemoteAddr
		if r.URL.Path == "/" {
			Redirect(w, r, "/foo", StatusFound) // which includes a body
		}
	}))

	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	_, err = io.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		t.Fatal(err)
	}

	var first string
	select {
	case first = <-saw:
	default:
		t.Fatal("server didn't see a request")
	}

	var second string
	select {
	case second = <-saw:
	default:
		t.Fatal("server didn't see a second request")
	}

	if first != second {
		t.Fatal("server saw different client ports before & after the redirect")
	}
}

// eofReaderFunc is an io.Reader that runs itself, and then returns io.EOF.
type eofReaderFunc func()

func (f eofReaderFunc) Read(p []byte) (n int, err error) {
	f()
	return 0, io.EOF
}

func TestReferer(t *testing.T) {
	tests := []struct {
		lastReq, newReq, explicitRef string // from -> to URLs, explicitly set Referer value
		want                         string
	}{
		// don't send user:
		{lastReq: "http://gopher@test.com", newReq: "http://link.com", want: "http://test.com"},
		{lastReq: "https://gopher@test.com", newReq: "https://link.com", want: "https://test.com"},

		// don't send a user and password:
		{lastReq: "http://gopher:go@test.com", newReq: "http://link.com", want: "http://test.com"},
		{lastReq: "https://gopher:go@test.com", newReq: "https://link.com", want: "https://test.com"},

		// nothing to do:
		{lastReq: "http://test.com", newReq: "http://link.com", want: "http://test.com"},
		{lastReq: "https://test.com", newReq: "https://link.com", want: "https://test.com"},

		// https to http doesn't send a referer:
		{lastReq: "https://test.com", newReq: "http://link.com", want: ""},
		{lastReq: "https://gopher:go@test.com", newReq: "http://link.com", want: ""},

		// https to http should remove an existing referer:
		{lastReq: "https://test.com", newReq: "http://link.com", explicitRef: "https://foo.com", want: ""},
		{lastReq: "https://gopher:go@test.com", newReq: "http://link.com", explicitRef: "https://foo.com", want: ""},

		// don't override an existing referer:
		{lastReq: "https://test.com", newReq: "https://link.com", explicitRef: "https://foo.com", want: "https://foo.com"},
		{lastReq: "https://gopher:go@test.com", newReq: "https://link.com", explicitRef: "https://foo.com", want: "https://foo.com"},
	}
	for _, tt := range tests {
		l, err := url.Parse(tt.lastReq)
		if err != nil {
			t.Fatal(err)
		}
		n, err := url.Parse(tt.newReq)
		if err != nil {
			t.Fatal(err)
		}
		r := ExportRefererForURL(l, n, tt.explicitRef)
		if r != tt.want {
			t.Errorf("refererForURL(%q, %q) = %q; want %q", tt.lastReq, tt.newReq, r, tt.want)
		}
	}
}

// issue15577Tripper returns a Response with a redirect response
// header and doesn't populate its Response.Request field.
type issue15577Tripper struct{}

func (issue15577Tripper) RoundTrip(*Request) (*Response, error) {
	resp := &Response{
		StatusCode: 303,
		Header:     map[string][]string{"Location": {"http://www.example.com/"}},
		Body:       io.NopCloser(strings.NewReader("")),
	}
	return resp, nil
}

// Issue 15577: don't assume the roundtripper's response populates its Request field.
func TestClientRedirectResponseWithoutRequest(t *testing.T) {
	c := &Client{
		CheckRedirect: func(*Request, []*Request) error { return fmt.Errorf("no redirects!") },
		Transport:     issue15577Tripper{},
	}
	// Check that this doesn't crash:
	c.Get("http://dummy.tld")
}

// Issue 4800: copy (some) headers when Client follows a redirect.
// Issue 35104: Since both URLs have the same host (localhost)
// but different ports, sensitive headers like Cookie and Authorization
// are preserved.
func TestClientCopyHeadersOnRedirect(t *testing.T) { run(t, testClientCopyHeadersOnRedirect) }
func testClientCopyHeadersOnRedirect(t *testing.T, mode testMode) {
	const (
		ua   = "some-agent/1.2"
		xfoo = "foo-val"
	)
	var ts2URL string
	ts1 := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		want := Header{
			"User-Agent":      []string{ua},
			"X-Foo":           []string{xfoo},
			"Referer":         []string{ts2URL},
			"Accept-Encoding": []string{"gzip"},
			"Cookie":          []string{"foo=bar"},
			"Authorization":   []string{"secretpassword"},
		}
		if !reflect.DeepEqual(r.Header, want) {
			t.Errorf("Request.Header = %#v; want %#v", r.Header, want)
		}
		if t.Failed() {
			w.Header().Set("Result", "got errors")
		} else {
			w.Header().Set("Result", "ok")
		}
	})).ts
	ts2 := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		Redirect(w, r, ts1.URL, StatusFound)
	})).ts
	ts2URL = ts2.URL

	c := ts1.Client()
	c.CheckRedirect = func(r *Request, via []*Request) error {
		want := Header{
			"User-Agent":    []string{ua},
			"X-Foo":         []string{xfoo},
			"Referer":       []string{ts2URL},
			"Cookie":        []string{"foo=bar"},
			"Authorization": []string{"secretpassword"},
		}
		if !reflect.DeepEqual(r.Header, want) {
			t.Errorf("CheckRedirect Request.Header = %#v; want %#v", r.Header, want)
		}
		return nil
	}

	req, _ := NewRequest("GET", ts2.URL, nil)
	req.Header.Add("User-Agent", ua)
	req.Header.Add("X-Foo", xfoo)
	req.Header.Add("Cookie", "foo=bar")
	req.Header.Add("Authorization", "secretpassword")
	res, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		t.Fatal(res.Status)
	}
	if got := res.Header.Get("Result"); got != "ok" {
		t.Errorf("result = %q; want ok", got)
	}
}

// Issue 22233: copy host when Client follows a relative redirect.
func TestClientCopyHostOnRedirect(t *testing.T) { run(t, testClientCopyHostOnRedirect) }
func testClientCopyHostOnRedirect(t *testing.T, mode testMode) {
	// Virtual hostname: should not receive any request.
	virtual := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		t.Errorf("Virtual host received request %v", r.URL)
		w.WriteHeader(403)
		io.WriteString(w, "should not see this response")
	})).ts
	defer virtual.Close()
	virtualHost := strings.TrimPrefix(virtual.URL, "http://")
	virtualHost = strings.TrimPrefix(virtualHost, "https://")
	t.Logf("Virtual host is %v", virtualHost)

	// Actual hostname: should not receive any request.
	const wantBody = "response body"
	var tsURL string
	var tsHost string
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		switch r.URL.Path {
		case "/":
			// Relative redirect.
			if r.Host != virtualHost {
				t.Errorf("Serving /: Request.Host = %#v; want %#v", r.Host, virtualHost)
				w.WriteHeader(404)
				return
			}
			w.Header().Set("Location", "/hop")
			w.WriteHeader(302)
		case "/hop":
			// Absolute redirect.
			if r.Host != virtualHost {
				t.Errorf("Serving /hop: Request.Host = %#v; want %#v", r.Host, virtualHost)
				w.WriteHeader(404)
				return
			}
			w.Header().Set("Location", tsURL+"/final")
			w.WriteHeader(302)
		case "/final":
			if r.Host != tsHost {
				t.Errorf("Serving /final: Request.Host = %#v; want %#v", r.Host, tsHost)
				w.WriteHeader(404)
				return
			}
			w.WriteHeader(200)
			io.WriteString(w, wantBody)
		default:
			t.Errorf("Serving unexpected path %q", r.URL.Path)
			w.WriteHeader(404)
		}
	})).ts
	tsURL = ts.URL
	tsHost = strings.TrimPrefix(ts.URL, "http://")
	tsHost = strings.TrimPrefix(tsHost, "https://")
	t.Logf("Server host is %v", tsHost)

	c := ts.Client()
	req, _ := NewRequest("GET", ts.URL, nil)
	req.Host = virtualHost
	resp, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatal(resp.Status)
	}
	if got, err := io.ReadAll(resp.Body); err != nil || string(got) != wantBody {
		t.Errorf("body = %q; want %q", got, wantBody)
	}
}

// Issue 17494: cookies should be altered when Client follows redirects.
func TestClientAltersCookiesOnRedirect(t *testing.T) { run(t, testClientAltersCookiesOnRedirect) }
func testClientAltersCookiesOnRedirect(t *testing.T, mode testMode) {
	cookieMap := func(cs []*Cookie) map[string][]string {
		m := make(map[string][]string)
		for _, c := range cs {
			m[c.Name] = append(m[c.Name], c.Value)
		}
		return m
	}

	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		var want map[string][]string
		got := cookieMap(r.Cookies())

		c, _ := r.Cookie("Cycle")
		switch c.Value {
		case "0":
			want = map[string][]string{
				"Cookie1": {"OldValue1a", "OldValue1b"},
				"Cookie2": {"OldValue2"},
				"Cookie3": {"OldValue3a", "OldValue3b"},
				"Cookie4": {"OldValue4"},
				"Cycle":   {"0"},
			}
			SetCookie(w, &Cookie{Name: "Cycle", Value: "1", Path: "/"})
			SetCookie(w, &Cookie{Name: "Cookie2", Path: "/", MaxAge: -1}) // Delete cookie from Header
			Redirect(w, r, "/", StatusFound)
		case "1":
			want = map[string][]string{
				"Cookie1": {"OldValue1a", "OldValue1b"},
				"Cookie3": {"OldValue3a", "OldValue3b"},
				"Cookie4": {"OldValue4"},
				"Cycle":   {"1"},
			}
			SetCookie(w, &Cookie{Name: "Cycle", Value: "2", Path: "/"})
			SetCookie(w, &Cookie{Name: "Cookie3", Value: "NewValue3", Path: "/"}) // Modify cookie in Header
			SetCookie(w, &Cookie{Name: "Cookie4", Value: "NewValue4", Path: "/"}) // Modify cookie in Jar
			Redirect(w, r, "/", StatusFound)
		case "2":
			want = map[string][]string{
				"Cookie1": {"OldValue1a", "OldValue1b"},
				"Cookie3": {"NewValue3"},
				"Cookie4": {"NewValue4"},
				"Cycle":   {"2"},
			}
			SetCookie(w, &Cookie{Name: "Cycle", Value: "3", Path: "/"})
			SetCookie(w, &Cookie{Name: "Cookie5", Value: "NewValue5", Path: "/"}) // Insert cookie into Jar
			Redirect(w, r, "/", StatusFound)
		case "3":
			want = map[string][]string{
				"Cookie1": {"OldValue1a", "OldValue1b"},
				"Cookie3": {"NewValue3"},
				"Cookie4": {"NewValue4"},
				"Cookie5": {"NewValue5"},
				"Cycle":   {"3"},
			}
			// Don't redirect to ensure the loop ends.
		default:
			t.Errorf("unexpected redirect cycle")
			return
		}

		if !reflect.DeepEqual(got, want) {
			t.Errorf("redirect %s, Cookie = %v, want %v", c.Value, got, want)
		}
	})).ts

	jar, _ := cookiejar.New(nil)
	c := ts.Client()
	c.Jar = jar

	u, _ := url.Parse(ts.URL)
	req, _ := NewRequest("GET", ts.URL, nil)
	req.AddCookie(&Cookie{Name: "Cookie1", Value: "OldValue1a"})
	req.AddCookie(&Cookie{Name: "Cookie1", Value: "OldValue1b"})
	req.AddCookie(&Cookie{Name: "Cookie2", Value: "OldValue2"})
	req.AddCookie(&Cookie{Name: "Cookie3", Value: "OldValue3a"})
	req.AddCookie(&Cookie{Name: "Cookie3", Value: "OldValue3b"})
	jar.SetCookies(u, []*Cookie{{Name: "Cookie4", Value: "OldValue4", Path: "/"}})
	jar.SetCookies(u, []*Cookie{{Name: "Cycle", Value: "0", Path: "/"}})
	res, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		t.Fatal(res.Status)
	}
}

// Part of Issue 4800
func TestShouldCopyHeaderOnRedirect(t *testing.T) {
	tests := []struct {
		header     string
		initialURL string
		destURL    string
		want       bool
	}{
		{"User-Agent", "http://foo.com/", "http://bar.com/", true},
		{"X-Foo", "http://foo.com/", "http://bar.com/", true},

		// Sensitive headers:
		{"cookie", "http://foo.com/", "http://bar.com/", false},
		{"cookie2", "http://foo.com/", "http://bar.com/", false},
		{"authorization", "http://foo.com/", "http://bar.com/", false},
		{"authorization", "http://foo.com/", "https://foo.com/", true},
		{"authorization", "http://foo.com:1234/", "http://foo.com:4321/", true},
		{"www-authenticate", "http://foo.com/", "http://bar.com/", false},
		{"authorization", "http://foo.com/", "http://[::1%25.foo.com]/", false},

		// But subdomains should work:
		{"www-authenticate", "http://foo.com/", "http://foo.com/", true},
		{"www-authenticate", "http://foo.com/", "http://sub.foo.com/", true},
		{"www-authenticate", "http://foo.com/", "http://notfoo.com/", false},
		{"www-authenticate", "http://foo.com/", "https://foo.com/", true},
		{"www-authenticate", "http://foo.com:80/", "http://foo.com/", true},
		{"www-authenticate", "http://foo.com:80/", "http://sub.foo.com/", true},
		{"www-authenticate", "http://foo.com:443/", "https://foo.com/", true},
		{"www-authenticate", "http://foo.com:443/", "https://sub.foo.com/", true},
		{"www-authenticate", "http://foo.com:1234/", "http://foo.com/", true},

		{"authorization", "http://foo.com/", "http://foo.com/", true},
		{"authorization", "http://foo.com/", "http://sub.foo.com/", true},
		{"authorization", "http://foo.com/", "http://notfoo.com/", false},
		{"authorization", "http://foo.com/", "https://foo.com/", true},
		{"authorization", "http://foo.com:80/", "http://foo.com/", true},
		{"authorization", "http://foo.com:80/", "http://sub.foo.com/", true},
		{"authorization", "http://foo.com:443/", "https://foo.com/", true},
		{"authorization", "http://foo.com:443/", "https://sub.foo.com/", true},
		{"authorization", "http://foo.com:1234/", "http://foo.com/", true},
	}
	for i, tt := range tests {
		u0, err := url.Parse(tt.initialURL)
		if err != nil {
			t.Errorf("%d. initial URL %q parse error: %v", i, tt.initialURL, err)
			continue
		}
		u1, err := url.Parse(tt.destURL)
		if err != nil {
			t.Errorf("%d. dest URL %q parse error: %v", i, tt.destURL, err)
			continue
		}
		got := Export_shouldCopyHeaderOnRedirect(tt.header, u0, u1)
		if got != tt.want {
			t.Errorf("%d. shouldCopyHeaderOnRedirect(%q, %q => %q) = %v; want %v",
				i, tt.header, tt.initialURL, tt.destURL, got, tt.want)
		}
	}
}

func TestClientRedirectTypes(t *testing.T) { run(t, testClientRedirectTypes) }
func testClientRedirectTypes(t *testing.T, mode testMode) {
	tests := [...]struct {
		method       string
		serverStatus int
		wantMethod   string // desired subsequent client method
	}{
		0: {method: "POST", serverStatus: 301, wantMethod: "GET"},
		1: {method: "POST", serverStatus: 302, wantMethod: "GET"},
		2: {method: "POST", serverStatus: 303, wantMethod: "GET"},
		3: {method: "POST", serverStatus: 307, wantMethod: "POST"},
		4: {method: "POST", serverStatus: 308, wantMethod: "POST"},

		5: {method: "HEAD", serverStatus: 301, wantMethod: "HEAD"},
		6: {method: "HEAD", serverStatus: 302, wantMethod: "HEAD"},
		7: {method: "HEAD", serverStatus: 303, wantMethod: "HEAD"},
		8: {method: "HEAD", serverStatus: 307, wantMethod: "HEAD"},
		9: {method: "HEAD", serverStatus: 308, wantMethod: "HEAD"},

		10: {method: "GET", serverStatus: 301, wantMethod: "GET"},
		11: {method: "GET", serverStatus: 302, wantMethod: "GET"},
		12: {method: "GET", serverStatus: 303, wantMethod: "GET"},
		13: {method: "GET", serverStatus: 307, wantMethod: "GET"},
		14: {method: "GET", serverStatus: 308, wantMethod: "GET"},

		15: {method: "DELETE", serverStatus: 301, wantMethod: "GET"},
		16: {method: "DELETE", serverStatus: 302, wantMethod: "GET"},
		17: {method: "DELETE", serverStatus: 303, wantMethod: "GET"},
		18: {method: "DELETE", serverStatus: 307, wantMethod: "DELETE"},
		19: {method: "DELETE", serverStatus: 308, wantMethod: "DELETE"},

		20: {method: "PUT", serverStatus: 301, wantMethod: "GET"},
		21: {method: "PUT", serverStatus: 302, wantMethod: "GET"},
		22: {method: "PUT", serverStatus: 303, wantMethod: "GET"},
		23: {method: "PUT", serverStatus: 307, wantMethod: "PUT"},
		24: {method: "PUT", serverStatus: 308, wantMethod: "PUT"},

		25: {method: "MADEUPMETHOD", serverStatus: 301, wantMethod: "GET"},
		26: {method: "MADEUPMETHOD", serverStatus: 302, wantMethod: "GET"},
		27: {method: "MADEUPMETHOD", serverStatus: 303, wantMethod: "GET"},
		28: {method: "MADEUPMETHOD", serverStatus: 307, wantMethod: "MADEUPMETHOD"},
		29: {method: "MADEUPMETHOD", serverStatus: 308, wantMethod: "MADEUPMETHOD"},
	}

	handlerc := make(chan HandlerFunc, 1)

	ts := newClientServerTest(t, mode, HandlerFunc(func(rw ResponseWriter, req *Request) {
		h := <-handlerc
		h(rw, req)
	})).ts

	c := ts.Client()
	for i, tt := range tests {
		handlerc <- func(w ResponseWriter, r *Request) {
			w.Header().Set("Location", ts.URL)
			w.WriteHeader(tt.serverStatus)
		}

		req, err := NewRequest(tt.method, ts.URL, nil)
		if err != nil {
			t.Errorf("#%d: NewRequest: %v", i, err)
			continue
		}

		c.CheckRedirect = func(req *Request, via []*Request) error {
			if got, want := req.Method, tt.wantMethod; got != want {
				return fmt.Errorf("#%d: got next method %q; want %q", i, got, want)
			}
			handlerc <- func(rw ResponseWriter, req *Request) {
				// TODO: Check that the body is valid when we do 307 and 308 support
			}
			return nil
		}

		res, err := c.Do(req)
		if err != nil {
			t.Errorf("#%d: Response: %v", i, err)
			continue
		}

		res.Body.Close()
	}
}

// issue18239Body is an io.ReadCloser for TestTransportBodyReadError.
// Its Read returns readErr and increments *readCalls atomically.
// Its Close returns nil and increments *closeCalls atomically.
type issue18239Body struct {
	readCalls  *int32
	closeCalls *int32
	readErr    error
}

func (b issue18239Body) Read([]byte) (int, error) {
	atomic.AddInt32(b.readCalls, 1)
	return 0, b.readErr
}

func (b issue18239Body) Close() error {
	atomic.AddInt32(b.closeCalls, 1)
	return nil
}

// Issue 18239: make sure the Transport doesn't retry requests with bodies
// if Request.GetBody is not defined.
func TestTransportBodyReadError(t *testing.T) { run(t, testTransportBodyReadError) }
func testTransportBodyReadError(t *testing.T, mode testMode) {
	ts := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		if r.URL.Path == "/ping" {
			return
		}
		buf := make([]byte, 1)
		n, err := r.Body.Read(buf)
		w.Header().Set("X-Body-Read", fmt.Sprintf("%v, %v", n, err))
	})).ts
	c := ts.Client()
	tr := c.Transport.(*Transport)

	// Do one initial successful request to create an idle TCP connection
	// for the subsequent request to reuse. (The Transport only retries
	// requests on reused connections.)
	res, err := c.Get(ts.URL + "/ping")
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()

	var readCallsAtomic int32
	var closeCallsAtomic int32 // atomic
	someErr := errors.New("some body read error")
	body := issue18239Body{&readCallsAtomic, &closeCallsAtomic, someErr}

	req, err := NewRequest("POST", ts.URL, body)
	if err != nil {
		t.Fatal(err)
	}
	req = req.WithT(t)
	_, err = tr.RoundTrip(req)
	if err != someErr {
		t.Errorf("Got error: %v; want Request.Body read error: %v", err, someErr)
	}

	// And verify that our Body wasn't used multiple times, which
	// would indicate retries. (as it buggily was during part of
	// Go 1.8's dev cycle)
	readCalls := atomic.LoadInt32(&readCallsAtomic)
	closeCalls := atomic.LoadInt32(&closeCallsAtomic)
	if readCalls != 1 {
		t.Errorf("read calls = %d; want 1", readCalls)
	}
	if closeCalls != 1 {
		t.Errorf("close calls = %d; want 1", closeCalls)
	}
}

type roundTripperWithoutCloseIdle struct{}

func (roundTripperWithoutCloseIdle) RoundTrip(*Request) (*Response, error) { panic("unused") }

type roundTripperWithCloseIdle func() // underlying func is CloseIdleConnections func

func (roundTripperWithCloseIdle) RoundTrip(*Request) (*Response, error) { panic("unused") }
func (f roundTripperWithCloseIdle) CloseIdleConnections()               { f() }

func TestClientCloseIdleConnections(t *testing.T) {
	c := &Client{Transport: roundTripperWithoutCloseIdle{}}
	c.CloseIdleConnections() // verify we don't crash at least

	closed := false
	var tr RoundTripper = roundTripperWithCloseIdle(func() {
		closed = true
	})
	c = &Client{Transport: tr}
	c.CloseIdleConnections()
	if !closed {
		t.Error("not closed")
	}
}

type testRoundTripper func(*Request) (*Response, error)

func (t testRoundTripper) RoundTrip(req *Request) (*Response, error) {
	return t(req)
}

func TestClientPropagatesTimeoutToContext(t *testing.T) {
	c := &Client{
		Timeout: 5 * time.Second,
		Transport: testRoundTripper(func(req *Request) (*Response, error) {
			ctx := req.Context()
			deadline, ok := ctx.Deadline()
			if !ok {
				t.Error("no deadline")
			} else {
				t.Logf("deadline in %v", deadline.Sub(time.Now()).Round(time.Second/10))
			}
			return nil, errors.New("not actually making a request")
		}),
	}
	c.Get("https://example.tld/")
}

// Issue 33545: lock-in the behavior promised by Client.Do's
// docs about request cancellation vs timing out.
func TestClientDoCanceledVsTimeout(t *testing.T) { run(t, testClientDoCanceledVsTimeout) }
func testClientDoCanceledVsTimeout(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Write([]byte("Hello, World!"))
	}))

	cases := []string{"timeout", "canceled"}

	for _, name := range cases {
		t.Run(name, func(t *testing.T) {
			var ctx context.Context
			var cancel func()
			if name == "timeout" {
				ctx, cancel = context.WithTimeout(context.Background(), -time.Nanosecond)
			} else {
				ctx, cancel = context.WithCancel(context.Background())
				cancel()
			}
			defer cancel()

			req, _ := NewRequestWithContext(ctx, "GET", cst.ts.URL, nil)
			_, err := cst.c.Do(req)
			if err == nil {
				t.Fatal("Unexpectedly got a nil error")
			}

			ue := err.(*url.Error)

			var wantIsTimeout bool
			var wantErr error = context.Canceled
			if name == "timeout" {
				wantErr = context.DeadlineExceeded
				wantIsTimeout = true
			}
			if g, w := ue.Timeout(), wantIsTimeout; g != w {
				t.Fatalf("url.Timeout() = %t, want %t", g, w)
			}
			if g, w := ue.Err, wantErr; g != w {
				t.Errorf("url.Error.Err = %v; want %v", g, w)
			}
			if got := errors.Is(err, context.DeadlineExceeded); got != wantIsTimeout {
				t.Errorf("errors.Is(err, context.DeadlineExceeded) = %v, want %v", got, wantIsTimeout)
			}
		})
	}
}

type nilBodyRoundTripper struct{}

func (nilBodyRoundTripper) RoundTrip(req *Request) (*Response, error) {
	return &Response{
		StatusCode: StatusOK,
		Status:     StatusText(StatusOK),
		Body:       nil,
		Request:    req,
	}, nil
}

func TestClientPopulatesNilResponseBody(t *testing.T) {
	c := &Client{Transport: nilBodyRoundTripper{}}

	resp, err := c.Get("http://localhost/anything")
	if err != nil {
		t.Fatalf("Client.Get rejected Response with nil Body: %v", err)
	}

	if resp.Body == nil {
		t.Fatalf("Client failed to provide a non-nil Body as documented")
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			t.Fatalf("error from Close on substitute Response.Body: %v", err)
		}
	}()

	if b, err := io.ReadAll(resp.Body); err != nil {
		t.Errorf("read error from substitute Response.Body: %v", err)
	} else if len(b) != 0 {
		t.Errorf("substitute Response.Body was unexpectedly non-empty: %q", b)
	}
}

// Issue 40382: Client calls Close multiple times on Request.Body.
func TestClientCallsCloseOnlyOnce(t *testing.T) { run(t, testClientCallsCloseOnlyOnce) }
func testClientCallsCloseOnlyOnce(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.WriteHeader(StatusNoContent)
	}))

	// Issue occurred non-deterministically: needed to occur after a successful
	// write (into TCP buffer) but before end of body.
	for i := 0; i < 50 && !t.Failed(); i++ {
		body := &issue40382Body{t: t, n: 300000}
		req, err := NewRequest(MethodPost, cst.ts.URL, body)
		if err != nil {
			t.Fatal(err)
		}
		resp, err := cst.tr.RoundTrip(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
	}
}

// issue40382Body is an io.ReadCloser for TestClientCallsCloseOnlyOnce.
// Its Read reads n bytes before returning io.EOF.
// Its Close returns nil but fails the test if called more than once.
type issue40382Body struct {
	t                *testing.T
	n                int
	closeCallsAtomic int32
}

func (b *issue40382Body) Read(p []byte) (int, error) {
	switch {
	case b.n == 0:
		return 0, io.EOF
	case b.n < len(p):
		p = p[:b.n]
		fallthrough
	default:
		for i := range p {
			p[i] = 'x'
		}
		b.n -= len(p)
		return len(p), nil
	}
}

func (b *issue40382Body) Close() error {
	if atomic.AddInt32(&b.closeCallsAtomic, 1) == 2 {
		b.t.Error("Body closed more than once")
	}
	return nil
}

func TestProbeZeroLengthBody(t *testing.T) { run(t, testProbeZeroLengthBody) }
func testProbeZeroLengthBody(t *testing.T, mode testMode) {
	reqc := make(chan struct{})
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		close(reqc)
		if _, err := io.Copy(w, r.Body); err != nil {
			t.Errorf("error copying request body: %v", err)
		}
	}))

	bodyr, bodyw := io.Pipe()
	var gotBody string
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		req, _ := NewRequest("GET", cst.ts.URL, bodyr)
		res, err := cst.c.Do(req)
		b, err := io.ReadAll(res.Body)
		if err != nil {
			t.Error(err)
		}
		gotBody = string(b)
	}()

	select {
	case <-reqc:
		// Request should be sent after trying to probe the request body for 200ms.
	case <-time.After(60 * time.Second):
		t.Errorf("request not sent after 60s")
	}

	// Write the request body and wait for the request to complete.
	const content = "body"
	bodyw.Write([]byte(content))
	bodyw.Close()
	wg.Wait()
	if gotBody != content {
		t.Fatalf("server got body %q, want %q", gotBody, content)
	}
}
