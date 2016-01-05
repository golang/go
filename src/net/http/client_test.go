// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for client.go

package http_test

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	. "net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

var robotsTxtHandler = HandlerFunc(func(w ResponseWriter, r *Request) {
	w.Header().Set("Last-Modified", "sometime")
	fmt.Fprintf(w, "User-agent: go\nDisallow: /something/")
})

// pedanticReadAll works like ioutil.ReadAll but additionally
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

type chanWriter chan string

func (w chanWriter) Write(p []byte) (n int, err error) {
	w <- string(p)
	return len(p), nil
}

func TestClient(t *testing.T) {
	defer afterTest(t)
	ts := httptest.NewServer(robotsTxtHandler)
	defer ts.Close()

	r, err := Get(ts.URL)
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

func TestClientHead_h1(t *testing.T) { testClientHead(t, h1Mode) }
func TestClientHead_h2(t *testing.T) { testClientHead(t, h2Mode) }

func testClientHead(t *testing.T, h2 bool) {
	defer afterTest(t)
	cst := newClientServerTest(t, h2, robotsTxtHandler)
	defer cst.close()

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
	bodyb, err := ioutil.ReadAll(tr.req.Body)
	if err != nil {
		t.Fatalf("ReadAll on req.Body: %v", err)
	}
	if g := string(bodyb); g != expectedBody && g != expectedBody1 {
		t.Errorf("got body %q, want %q or %q", g, expectedBody, expectedBody1)
	}
}

func TestClientRedirects(t *testing.T) {
	defer afterTest(t)
	var ts *httptest.Server
	ts = httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		n, _ := strconv.Atoi(r.FormValue("n"))
		// Test Referer header. (7 is arbitrary position to test at)
		if n == 7 {
			if g, e := r.Referer(), ts.URL+"/?n=6"; e != g {
				t.Errorf("on request ?n=7, expected referer of %q; got %q", e, g)
			}
		}
		if n < 15 {
			Redirect(w, r, fmt.Sprintf("/?n=%d", n+1), StatusFound)
			return
		}
		fmt.Fprintf(w, "n=%d", n)
	}))
	defer ts.Close()

	c := &Client{}
	_, err := c.Get(ts.URL)
	if e, g := "Get /?n=10: stopped after 10 redirects", fmt.Sprintf("%v", err); e != g {
		t.Errorf("with default client Get, expected error %q, got %q", e, g)
	}

	// HEAD request should also have the ability to follow redirects.
	_, err = c.Head(ts.URL)
	if e, g := "Head /?n=10: stopped after 10 redirects", fmt.Sprintf("%v", err); e != g {
		t.Errorf("with default client Head, expected error %q, got %q", e, g)
	}

	// Do should also follow redirects.
	greq, _ := NewRequest("GET", ts.URL, nil)
	_, err = c.Do(greq)
	if e, g := "Get /?n=10: stopped after 10 redirects", fmt.Sprintf("%v", err); e != g {
		t.Errorf("with default client Do, expected error %q, got %q", e, g)
	}

	// Requests with an empty Method should also redirect (Issue 12705)
	greq.Method = ""
	_, err = c.Do(greq)
	if e, g := "Get /?n=10: stopped after 10 redirects", fmt.Sprintf("%v", err); e != g {
		t.Errorf("with default client Do and empty Method, expected error %q, got %q", e, g)
	}

	var checkErr error
	var lastVia []*Request
	c = &Client{CheckRedirect: func(_ *Request, via []*Request) error {
		lastVia = via
		return checkErr
	}}
	res, err := c.Get(ts.URL)
	if err != nil {
		t.Fatalf("Get error: %v", err)
	}
	res.Body.Close()
	finalUrl := res.Request.URL.String()
	if e, g := "<nil>", fmt.Sprintf("%v", err); e != g {
		t.Errorf("with custom client, expected error %q, got %q", e, g)
	}
	if !strings.HasSuffix(finalUrl, "/?n=15") {
		t.Errorf("expected final url to end in /?n=15; got url %q", finalUrl)
	}
	if e, g := 15, len(lastVia); e != g {
		t.Errorf("expected lastVia to have contained %d elements; got %d", e, g)
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

func TestPostRedirects(t *testing.T) {
	defer afterTest(t)
	var log struct {
		sync.Mutex
		bytes.Buffer
	}
	var ts *httptest.Server
	ts = httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		log.Lock()
		fmt.Fprintf(&log.Buffer, "%s %s ", r.Method, r.RequestURI)
		log.Unlock()
		if v := r.URL.Query().Get("code"); v != "" {
			code, _ := strconv.Atoi(v)
			if code/100 == 3 {
				w.Header().Set("Location", ts.URL)
			}
			w.WriteHeader(code)
		}
	}))
	defer ts.Close()
	tests := []struct {
		suffix string
		want   int // response code
	}{
		{"/", 200},
		{"/?code=301", 301},
		{"/?code=302", 200},
		{"/?code=303", 200},
		{"/?code=404", 404},
	}
	for _, tt := range tests {
		res, err := Post(ts.URL+tt.suffix, "text/plain", strings.NewReader("Some content"))
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
	want := "POST / POST /?code=301 POST /?code=302 GET / POST /?code=303 GET / POST /?code=404 "
	if got != want {
		t.Errorf("Log differs.\n Got: %q\nWant: %q", got, want)
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

func TestRedirectCookiesJar(t *testing.T) {
	defer afterTest(t)
	var ts *httptest.Server
	ts = httptest.NewServer(echoCookiesRedirectHandler)
	defer ts.Close()
	c := &Client{
		Jar: new(TestJar),
	}
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

func TestJarCalls(t *testing.T) {
	defer afterTest(t)
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		pathSuffix := r.RequestURI[1:]
		if r.RequestURI == "/nosetcookie" {
			return // don't set cookies for this path
		}
		SetCookie(w, &Cookie{Name: "name" + pathSuffix, Value: "val" + pathSuffix})
		if r.RequestURI == "/" {
			Redirect(w, r, "http://secondhost.fake/secondpath", 302)
		}
	}))
	defer ts.Close()
	jar := new(RecordingJar)
	c := &Client{
		Jar: jar,
		Transport: &Transport{
			Dial: func(_ string, _ string) (net.Conn, error) {
				return net.Dial("tcp", ts.Listener.Addr().String())
			},
		},
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

func (j *RecordingJar) logf(format string, args ...interface{}) {
	j.mu.Lock()
	defer j.mu.Unlock()
	fmt.Fprintf(&j.log, format, args...)
}

func TestStreamingGet_h1(t *testing.T) { testStreamingGet(t, h1Mode) }
func TestStreamingGet_h2(t *testing.T) { testStreamingGet(t, h2Mode) }

func testStreamingGet(t *testing.T, h2 bool) {
	defer afterTest(t)
	say := make(chan string)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.(Flusher).Flush()
		for str := range say {
			w.Write([]byte(str))
			w.(Flusher).Flush()
		}
	}))
	defer cst.close()

	c := cst.c
	res, err := c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	var buf [10]byte
	for _, str := range []string{"i", "am", "also", "known", "as", "comet"} {
		say <- str
		n, err := io.ReadFull(res.Body, buf[0:len(str)])
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
func TestClientWrites(t *testing.T) {
	defer afterTest(t)
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
	}))
	defer ts.Close()

	writes := 0
	dialer := func(netz string, addr string) (net.Conn, error) {
		c, err := net.Dial(netz, addr)
		if err == nil {
			c = &writeCountingConn{c, &writes}
		}
		return c, err
	}
	c := &Client{Transport: &Transport{Dial: dialer}}

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
	defer afterTest(t)
	ts := httptest.NewTLSServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Write([]byte("Hello"))
	}))
	errc := make(chanWriter, 10) // but only expecting 1
	ts.Config.ErrorLog = log.New(errc, "", 0)
	defer ts.Close()

	// TODO(bradfitz): add tests for skipping hostname checks too?
	// would require a new cert for testing, and probably
	// redundant with these tests.
	for _, insecure := range []bool{true, false} {
		tr := &Transport{
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: insecure,
			},
		}
		defer tr.CloseIdleConnections()
		c := &Client{Transport: tr}
		res, err := c.Get(ts.URL)
		if (err == nil) != insecure {
			t.Errorf("insecure=%v: got unexpected err=%v", insecure, err)
		}
		if res != nil {
			res.Body.Close()
		}
	}

	select {
	case v := <-errc:
		if !strings.Contains(v, "TLS handshake error") {
			t.Errorf("expected an error log message containing 'TLS handshake error'; got %q", v)
		}
	case <-time.After(5 * time.Second):
		t.Errorf("timeout waiting for logged error")
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

func newTLSTransport(t *testing.T, ts *httptest.Server) *Transport {
	certs := x509.NewCertPool()
	for _, c := range ts.TLS.Certificates {
		roots, err := x509.ParseCertificates(c.Certificate[len(c.Certificate)-1])
		if err != nil {
			t.Fatalf("error parsing server's root cert: %v", err)
		}
		for _, root := range roots {
			certs.AddCert(root)
		}
	}
	return &Transport{
		TLSClientConfig: &tls.Config{RootCAs: certs},
	}
}

func TestClientWithCorrectTLSServerName(t *testing.T) {
	defer afterTest(t)

	const serverName = "example.com"
	ts := httptest.NewTLSServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		if r.TLS.ServerName != serverName {
			t.Errorf("expected client to set ServerName %q, got: %q", serverName, r.TLS.ServerName)
		}
	}))
	defer ts.Close()

	trans := newTLSTransport(t, ts)
	trans.TLSClientConfig.ServerName = serverName
	c := &Client{Transport: trans}
	if _, err := c.Get(ts.URL); err != nil {
		t.Fatalf("expected successful TLS connection, got error: %v", err)
	}
}

func TestClientWithIncorrectTLSServerName(t *testing.T) {
	defer afterTest(t)
	ts := httptest.NewTLSServer(HandlerFunc(func(w ResponseWriter, r *Request) {}))
	defer ts.Close()
	errc := make(chanWriter, 10) // but only expecting 1
	ts.Config.ErrorLog = log.New(errc, "", 0)

	trans := newTLSTransport(t, ts)
	trans.TLSClientConfig.ServerName = "badserver"
	c := &Client{Transport: trans}
	_, err := c.Get(ts.URL)
	if err == nil {
		t.Fatalf("expected an error")
	}
	if !strings.Contains(err.Error(), "127.0.0.1") || !strings.Contains(err.Error(), "badserver") {
		t.Errorf("wanted error mentioning 127.0.0.1 and badserver; got error: %v", err)
	}
	select {
	case v := <-errc:
		if !strings.Contains(v, "TLS handshake error") {
			t.Errorf("expected an error log message containing 'TLS handshake error'; got %q", v)
		}
	case <-time.After(5 * time.Second):
		t.Errorf("timeout waiting for logged error")
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
	defer afterTest(t)
	ts := httptest.NewTLSServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Write([]byte("Hello"))
	}))
	defer ts.Close()

	tr := newTLSTransport(t, ts)
	tr.TLSClientConfig.ServerName = "example.com" // one of httptest's Server cert names
	tr.Dial = func(netw, addr string) (net.Conn, error) {
		return net.Dial(netw, ts.Listener.Addr().String())
	}
	defer tr.CloseIdleConnections()
	c := &Client{Transport: tr}
	res, err := c.Get("https://some-other-host.tld/")
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
}

func TestResponseSetsTLSConnectionState(t *testing.T) {
	defer afterTest(t)
	ts := httptest.NewTLSServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Write([]byte("Hello"))
	}))
	defer ts.Close()

	tr := newTLSTransport(t, ts)
	tr.TLSClientConfig.CipherSuites = []uint16{tls.TLS_RSA_WITH_3DES_EDE_CBC_SHA}
	tr.Dial = func(netw, addr string) (net.Conn, error) {
		return net.Dial(netw, ts.Listener.Addr().String())
	}
	defer tr.CloseIdleConnections()
	c := &Client{Transport: tr}
	res, err := c.Get("https://example.com/")
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.TLS == nil {
		t.Fatal("Response didn't set TLS Connection State.")
	}
	if got, want := res.TLS.CipherSuite, tls.TLS_RSA_WITH_3DES_EDE_CBC_SHA; got != want {
		t.Errorf("TLS Cipher Suite = %d; want %d", got, want)
	}
}

// Check that an HTTPS client can interpret a particular TLS error
// to determine that the server is speaking HTTP.
// See golang.org/issue/11111.
func TestHTTPSClientDetectsHTTPServer(t *testing.T) {
	defer afterTest(t)
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {}))
	defer ts.Close()

	_, err := Get(strings.Replace(ts.URL, "http", "https", 1))
	if got := err.Error(); !strings.Contains(got, "HTTP response to HTTPS client") {
		t.Fatalf("error = %q; want error indicating HTTP response to HTTPS request", got)
	}
}

// Verify Response.ContentLength is populated. https://golang.org/issue/4126
func TestClientHeadContentLength_h1(t *testing.T) {
	testClientHeadContentLength(t, h1Mode)
}

func TestClientHeadContentLength_h2(t *testing.T) {
	testClientHeadContentLength(t, h2Mode)
}

func testClientHeadContentLength(t *testing.T, h2 bool) {
	defer afterTest(t)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		if v := r.FormValue("cl"); v != "" {
			w.Header().Set("Content-Length", v)
		}
	}))
	defer cst.close()
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
		bs, err := ioutil.ReadAll(res.Body)
		if err != nil {
			t.Fatal(err)
		}
		if len(bs) != 0 {
			t.Errorf("Unexpected content: %q", bs)
		}
	}
}

func TestEmptyPasswordAuth(t *testing.T) {
	defer afterTest(t)
	gopher := "gopher"
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
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
	}))
	defer ts.Close()
	c := &Client{}
	req, err := NewRequest("GET", ts.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.URL.User = url.User(gopher)
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

func TestClientTimeout_h1(t *testing.T) { testClientTimeout(t, h1Mode) }
func TestClientTimeout_h2(t *testing.T) { testClientTimeout(t, h2Mode) }

func testClientTimeout(t *testing.T, h2 bool) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	defer afterTest(t)
	sawRoot := make(chan bool, 1)
	sawSlow := make(chan bool, 1)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		if r.URL.Path == "/" {
			sawRoot <- true
			Redirect(w, r, "/slow", StatusFound)
			return
		}
		if r.URL.Path == "/slow" {
			w.Write([]byte("Hello"))
			w.(Flusher).Flush()
			sawSlow <- true
			time.Sleep(2 * time.Second)
			return
		}
	}))
	defer cst.close()
	const timeout = 500 * time.Millisecond
	cst.c.Timeout = timeout

	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}

	select {
	case <-sawRoot:
		// good.
	default:
		t.Fatal("handler never got / request")
	}

	select {
	case <-sawSlow:
		// good.
	default:
		t.Fatal("handler never got /slow request")
	}

	errc := make(chan error, 1)
	go func() {
		_, err := ioutil.ReadAll(res.Body)
		errc <- err
		res.Body.Close()
	}()

	const failTime = timeout * 2
	select {
	case err := <-errc:
		if err == nil {
			t.Fatal("expected error from ReadAll")
		}
		ne, ok := err.(net.Error)
		if !ok {
			t.Errorf("error value from ReadAll was %T; expected some net.Error", err)
		} else if !ne.Timeout() {
			t.Errorf("net.Error.Timeout = false; want true")
		}
		if got := ne.Error(); !strings.Contains(got, "Client.Timeout exceeded") {
			t.Errorf("error string = %q; missing timeout substring", got)
		}
	case <-time.After(failTime):
		t.Errorf("timeout after %v waiting for timeout of %v", failTime, timeout)
	}
}

func TestClientTimeout_Headers_h1(t *testing.T) { testClientTimeout_Headers(t, h1Mode) }
func TestClientTimeout_Headers_h2(t *testing.T) { testClientTimeout_Headers(t, h2Mode) }

// Client.Timeout firing before getting to the body
func testClientTimeout_Headers(t *testing.T, h2 bool) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	defer afterTest(t)
	donec := make(chan bool)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		<-donec
	}))
	defer cst.close()
	// Note that we use a channel send here and not a close.
	// The race detector doesn't know that we're waiting for a timeout
	// and thinks that the waitgroup inside httptest.Server is added to concurrently
	// with us closing it. If we timed out immediately, we could close the testserver
	// before we entered the handler. We're not timing out immediately and there's
	// no way we would be done before we entered the handler, but the race detector
	// doesn't know this, so synchronize explicitly.
	defer func() { donec <- true }()

	cst.c.Timeout = 500 * time.Millisecond
	_, err := cst.c.Get(cst.ts.URL)
	if err == nil {
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
	if got := ne.Error(); !strings.Contains(got, "Client.Timeout exceeded") {
		t.Errorf("error string = %q; missing timeout substring", got)
	}
}

func TestClientRedirectEatsBody_h1(t *testing.T) { testClientRedirectEatsBody(t, h1Mode) }
func TestClientRedirectEatsBody_h2(t *testing.T) { testClientRedirectEatsBody(t, h2Mode) }
func testClientRedirectEatsBody(t *testing.T, h2 bool) {
	defer afterTest(t)
	saw := make(chan string, 2)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		saw <- r.RemoteAddr
		if r.URL.Path == "/" {
			Redirect(w, r, "/foo", StatusFound) // which includes a body
		}
	}))
	defer cst.close()

	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	_, err = ioutil.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()

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
		lastReq, newReq string // from -> to URLs
		want            string
	}{
		// don't send user:
		{"http://gopher@test.com", "http://link.com", "http://test.com"},
		{"https://gopher@test.com", "https://link.com", "https://test.com"},

		// don't send a user and password:
		{"http://gopher:go@test.com", "http://link.com", "http://test.com"},
		{"https://gopher:go@test.com", "https://link.com", "https://test.com"},

		// nothing to do:
		{"http://test.com", "http://link.com", "http://test.com"},
		{"https://test.com", "https://link.com", "https://test.com"},

		// https to http doesn't send a referer:
		{"https://test.com", "http://link.com", ""},
		{"https://gopher:go@test.com", "http://link.com", ""},
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
		r := ExportRefererForURL(l, n)
		if r != tt.want {
			t.Errorf("refererForURL(%q, %q) = %q; want %q", tt.lastReq, tt.newReq, r, tt.want)
		}
	}
}
