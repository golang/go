// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests that use both the client & server, in both HTTP/1 and HTTP/2 mode.

package http_test

import (
	"bytes"
	"compress/gzip"
	"crypto/tls"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	. "net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type clientServerTest struct {
	t  *testing.T
	h2 bool
	h  Handler
	ts *httptest.Server
	tr *Transport
	c  *Client
}

func (t *clientServerTest) close() {
	t.tr.CloseIdleConnections()
	t.ts.Close()
}

func (t *clientServerTest) getURL(u string) string {
	res, err := t.c.Get(u)
	if err != nil {
		t.t.Fatal(err)
	}
	defer res.Body.Close()
	slurp, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.t.Fatal(err)
	}
	return string(slurp)
}

func (t *clientServerTest) scheme() string {
	if t.h2 {
		return "https"
	}
	return "http"
}

const (
	h1Mode = false
	h2Mode = true
)

var optQuietLog = func(ts *httptest.Server) {
	ts.Config.ErrorLog = quietLog
}

func newClientServerTest(t *testing.T, h2 bool, h Handler, opts ...interface{}) *clientServerTest {
	cst := &clientServerTest{
		t:  t,
		h2: h2,
		h:  h,
		tr: &Transport{},
	}
	cst.c = &Client{Transport: cst.tr}
	cst.ts = httptest.NewUnstartedServer(h)

	for _, opt := range opts {
		switch opt := opt.(type) {
		case func(*Transport):
			opt(cst.tr)
		case func(*httptest.Server):
			opt(cst.ts)
		default:
			t.Fatalf("unhandled option type %T", opt)
		}
	}

	if !h2 {
		cst.ts.Start()
		return cst
	}
	ExportHttp2ConfigureServer(cst.ts.Config, nil)
	cst.ts.TLS = cst.ts.Config.TLSConfig
	cst.ts.StartTLS()

	cst.tr.TLSClientConfig = &tls.Config{
		InsecureSkipVerify: true,
	}
	if err := ExportHttp2ConfigureTransport(cst.tr); err != nil {
		t.Fatal(err)
	}
	return cst
}

// Testing the newClientServerTest helper itself.
func TestNewClientServerTest(t *testing.T) {
	var got struct {
		sync.Mutex
		log []string
	}
	h := HandlerFunc(func(w ResponseWriter, r *Request) {
		got.Lock()
		defer got.Unlock()
		got.log = append(got.log, r.Proto)
	})
	for _, v := range [2]bool{false, true} {
		cst := newClientServerTest(t, v, h)
		if _, err := cst.c.Head(cst.ts.URL); err != nil {
			t.Fatal(err)
		}
		cst.close()
	}
	got.Lock() // no need to unlock
	if want := []string{"HTTP/1.1", "HTTP/2.0"}; !reflect.DeepEqual(got.log, want) {
		t.Errorf("got %q; want %q", got.log, want)
	}
}

func TestChunkedResponseHeaders_h1(t *testing.T) { testChunkedResponseHeaders(t, h1Mode) }
func TestChunkedResponseHeaders_h2(t *testing.T) { testChunkedResponseHeaders(t, h2Mode) }

func testChunkedResponseHeaders(t *testing.T, h2 bool) {
	defer afterTest(t)
	log.SetOutput(ioutil.Discard) // is noisy otherwise
	defer log.SetOutput(os.Stderr)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Length", "intentional gibberish") // we check that this is deleted
		w.(Flusher).Flush()
		fmt.Fprintf(w, "I am a chunked response.")
	}))
	defer cst.close()

	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatalf("Get error: %v", err)
	}
	defer res.Body.Close()
	if g, e := res.ContentLength, int64(-1); g != e {
		t.Errorf("expected ContentLength of %d; got %d", e, g)
	}
	wantTE := []string{"chunked"}
	if h2 {
		wantTE = nil
	}
	if !reflect.DeepEqual(res.TransferEncoding, wantTE) {
		t.Errorf("TransferEncoding = %v; want %v", res.TransferEncoding, wantTE)
	}
	if got, haveCL := res.Header["Content-Length"]; haveCL {
		t.Errorf("Unexpected Content-Length: %q", got)
	}
}

type reqFunc func(c *Client, url string) (*Response, error)

// h12Compare is a test that compares HTTP/1 and HTTP/2 behavior
// against each other.
type h12Compare struct {
	Handler            func(ResponseWriter, *Request)    // required
	ReqFunc            reqFunc                           // optional
	CheckResponse      func(proto string, res *Response) // optional
	EarlyCheckResponse func(proto string, res *Response) // optional; pre-normalize
	Opts               []interface{}
}

func (tt h12Compare) reqFunc() reqFunc {
	if tt.ReqFunc == nil {
		return (*Client).Get
	}
	return tt.ReqFunc
}

func (tt h12Compare) run(t *testing.T) {
	setParallel(t)
	cst1 := newClientServerTest(t, false, HandlerFunc(tt.Handler), tt.Opts...)
	defer cst1.close()
	cst2 := newClientServerTest(t, true, HandlerFunc(tt.Handler), tt.Opts...)
	defer cst2.close()

	res1, err := tt.reqFunc()(cst1.c, cst1.ts.URL)
	if err != nil {
		t.Errorf("HTTP/1 request: %v", err)
		return
	}
	res2, err := tt.reqFunc()(cst2.c, cst2.ts.URL)
	if err != nil {
		t.Errorf("HTTP/2 request: %v", err)
		return
	}

	if fn := tt.EarlyCheckResponse; fn != nil {
		fn("HTTP/1.1", res1)
		fn("HTTP/2.0", res2)
	}

	tt.normalizeRes(t, res1, "HTTP/1.1")
	tt.normalizeRes(t, res2, "HTTP/2.0")
	res1body, res2body := res1.Body, res2.Body

	eres1 := mostlyCopy(res1)
	eres2 := mostlyCopy(res2)
	if !reflect.DeepEqual(eres1, eres2) {
		t.Errorf("Response headers to handler differed:\nhttp/1 (%v):\n\t%#v\nhttp/2 (%v):\n\t%#v",
			cst1.ts.URL, eres1, cst2.ts.URL, eres2)
	}
	if !reflect.DeepEqual(res1body, res2body) {
		t.Errorf("Response bodies to handler differed.\nhttp1: %v\nhttp2: %v\n", res1body, res2body)
	}
	if fn := tt.CheckResponse; fn != nil {
		res1.Body, res2.Body = res1body, res2body
		fn("HTTP/1.1", res1)
		fn("HTTP/2.0", res2)
	}
}

func mostlyCopy(r *Response) *Response {
	c := *r
	c.Body = nil
	c.TransferEncoding = nil
	c.TLS = nil
	c.Request = nil
	return &c
}

type slurpResult struct {
	io.ReadCloser
	body []byte
	err  error
}

func (sr slurpResult) String() string { return fmt.Sprintf("body %q; err %v", sr.body, sr.err) }

func (tt h12Compare) normalizeRes(t *testing.T, res *Response, wantProto string) {
	if res.Proto == wantProto {
		res.Proto, res.ProtoMajor, res.ProtoMinor = "", 0, 0
	} else {
		t.Errorf("got %q response; want %q", res.Proto, wantProto)
	}
	slurp, err := ioutil.ReadAll(res.Body)

	res.Body.Close()
	res.Body = slurpResult{
		ReadCloser: ioutil.NopCloser(bytes.NewReader(slurp)),
		body:       slurp,
		err:        err,
	}
	for i, v := range res.Header["Date"] {
		res.Header["Date"][i] = strings.Repeat("x", len(v))
	}
	if res.Request == nil {
		t.Errorf("for %s, no request", wantProto)
	}
	if (res.TLS != nil) != (wantProto == "HTTP/2.0") {
		t.Errorf("TLS set = %v; want %v", res.TLS != nil, res.TLS == nil)
	}
}

// Issue 13532
func TestH12_HeadContentLengthNoBody(t *testing.T) {
	h12Compare{
		ReqFunc: (*Client).Head,
		Handler: func(w ResponseWriter, r *Request) {
		},
	}.run(t)
}

func TestH12_HeadContentLengthSmallBody(t *testing.T) {
	h12Compare{
		ReqFunc: (*Client).Head,
		Handler: func(w ResponseWriter, r *Request) {
			io.WriteString(w, "small")
		},
	}.run(t)
}

func TestH12_HeadContentLengthLargeBody(t *testing.T) {
	h12Compare{
		ReqFunc: (*Client).Head,
		Handler: func(w ResponseWriter, r *Request) {
			chunk := strings.Repeat("x", 512<<10)
			for i := 0; i < 10; i++ {
				io.WriteString(w, chunk)
			}
		},
	}.run(t)
}

func TestH12_200NoBody(t *testing.T) {
	h12Compare{Handler: func(w ResponseWriter, r *Request) {}}.run(t)
}

func TestH2_204NoBody(t *testing.T) { testH12_noBody(t, 204) }
func TestH2_304NoBody(t *testing.T) { testH12_noBody(t, 304) }
func TestH2_404NoBody(t *testing.T) { testH12_noBody(t, 404) }

func testH12_noBody(t *testing.T, status int) {
	h12Compare{Handler: func(w ResponseWriter, r *Request) {
		w.WriteHeader(status)
	}}.run(t)
}

func TestH12_SmallBody(t *testing.T) {
	h12Compare{Handler: func(w ResponseWriter, r *Request) {
		io.WriteString(w, "small body")
	}}.run(t)
}

func TestH12_ExplicitContentLength(t *testing.T) {
	h12Compare{Handler: func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Length", "3")
		io.WriteString(w, "foo")
	}}.run(t)
}

func TestH12_FlushBeforeBody(t *testing.T) {
	h12Compare{Handler: func(w ResponseWriter, r *Request) {
		w.(Flusher).Flush()
		io.WriteString(w, "foo")
	}}.run(t)
}

func TestH12_FlushMidBody(t *testing.T) {
	h12Compare{Handler: func(w ResponseWriter, r *Request) {
		io.WriteString(w, "foo")
		w.(Flusher).Flush()
		io.WriteString(w, "bar")
	}}.run(t)
}

func TestH12_Head_ExplicitLen(t *testing.T) {
	h12Compare{
		ReqFunc: (*Client).Head,
		Handler: func(w ResponseWriter, r *Request) {
			if r.Method != "HEAD" {
				t.Errorf("unexpected method %q", r.Method)
			}
			w.Header().Set("Content-Length", "1235")
		},
	}.run(t)
}

func TestH12_Head_ImplicitLen(t *testing.T) {
	h12Compare{
		ReqFunc: (*Client).Head,
		Handler: func(w ResponseWriter, r *Request) {
			if r.Method != "HEAD" {
				t.Errorf("unexpected method %q", r.Method)
			}
			io.WriteString(w, "foo")
		},
	}.run(t)
}

func TestH12_HandlerWritesTooLittle(t *testing.T) {
	h12Compare{
		Handler: func(w ResponseWriter, r *Request) {
			w.Header().Set("Content-Length", "3")
			io.WriteString(w, "12") // one byte short
		},
		CheckResponse: func(proto string, res *Response) {
			sr, ok := res.Body.(slurpResult)
			if !ok {
				t.Errorf("%s body is %T; want slurpResult", proto, res.Body)
				return
			}
			if sr.err != io.ErrUnexpectedEOF {
				t.Errorf("%s read error = %v; want io.ErrUnexpectedEOF", proto, sr.err)
			}
			if string(sr.body) != "12" {
				t.Errorf("%s body = %q; want %q", proto, sr.body, "12")
			}
		},
	}.run(t)
}

// Tests that the HTTP/1 and HTTP/2 servers prevent handlers from
// writing more than they declared. This test does not test whether
// the transport deals with too much data, though, since the server
// doesn't make it possible to send bogus data. For those tests, see
// transport_test.go (for HTTP/1) or x/net/http2/transport_test.go
// (for HTTP/2).
func TestH12_HandlerWritesTooMuch(t *testing.T) {
	h12Compare{
		Handler: func(w ResponseWriter, r *Request) {
			w.Header().Set("Content-Length", "3")
			w.(Flusher).Flush()
			io.WriteString(w, "123")
			w.(Flusher).Flush()
			n, err := io.WriteString(w, "x") // too many
			if n > 0 || err == nil {
				t.Errorf("for proto %q, final write = %v, %v; want 0, some error", r.Proto, n, err)
			}
		},
	}.run(t)
}

// Verify that both our HTTP/1 and HTTP/2 request and auto-decompress gzip.
// Some hosts send gzip even if you don't ask for it; see golang.org/issue/13298
func TestH12_AutoGzip(t *testing.T) {
	h12Compare{
		Handler: func(w ResponseWriter, r *Request) {
			if ae := r.Header.Get("Accept-Encoding"); ae != "gzip" {
				t.Errorf("%s Accept-Encoding = %q; want gzip", r.Proto, ae)
			}
			w.Header().Set("Content-Encoding", "gzip")
			gz := gzip.NewWriter(w)
			io.WriteString(gz, "I am some gzipped content. Go go go go go go go go go go go go should compress well.")
			gz.Close()
		},
	}.run(t)
}

func TestH12_AutoGzip_Disabled(t *testing.T) {
	h12Compare{
		Opts: []interface{}{
			func(tr *Transport) { tr.DisableCompression = true },
		},
		Handler: func(w ResponseWriter, r *Request) {
			fmt.Fprintf(w, "%q", r.Header["Accept-Encoding"])
			if ae := r.Header.Get("Accept-Encoding"); ae != "" {
				t.Errorf("%s Accept-Encoding = %q; want empty", r.Proto, ae)
			}
		},
	}.run(t)
}

// Test304Responses verifies that 304s don't declare that they're
// chunking in their response headers and aren't allowed to produce
// output.
func Test304Responses_h1(t *testing.T) { test304Responses(t, h1Mode) }
func Test304Responses_h2(t *testing.T) { test304Responses(t, h2Mode) }

func test304Responses(t *testing.T, h2 bool) {
	defer afterTest(t)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.WriteHeader(StatusNotModified)
		_, err := w.Write([]byte("illegal body"))
		if err != ErrBodyNotAllowed {
			t.Errorf("on Write, expected ErrBodyNotAllowed, got %v", err)
		}
	}))
	defer cst.close()
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.TransferEncoding) > 0 {
		t.Errorf("expected no TransferEncoding; got %v", res.TransferEncoding)
	}
	body, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Error(err)
	}
	if len(body) > 0 {
		t.Errorf("got unexpected body %q", string(body))
	}
}

func TestH12_ServerEmptyContentLength(t *testing.T) {
	h12Compare{
		Handler: func(w ResponseWriter, r *Request) {
			w.Header()["Content-Type"] = []string{""}
			io.WriteString(w, "<html><body>hi</body></html>")
		},
	}.run(t)
}

func TestH12_RequestContentLength_Known_NonZero(t *testing.T) {
	h12requestContentLength(t, func() io.Reader { return strings.NewReader("FOUR") }, 4)
}

func TestH12_RequestContentLength_Known_Zero(t *testing.T) {
	h12requestContentLength(t, func() io.Reader { return nil }, 0)
}

func TestH12_RequestContentLength_Unknown(t *testing.T) {
	h12requestContentLength(t, func() io.Reader { return struct{ io.Reader }{strings.NewReader("Stuff")} }, -1)
}

func h12requestContentLength(t *testing.T, bodyfn func() io.Reader, wantLen int64) {
	h12Compare{
		Handler: func(w ResponseWriter, r *Request) {
			w.Header().Set("Got-Length", fmt.Sprint(r.ContentLength))
			fmt.Fprintf(w, "Req.ContentLength=%v", r.ContentLength)
		},
		ReqFunc: func(c *Client, url string) (*Response, error) {
			return c.Post(url, "text/plain", bodyfn())
		},
		CheckResponse: func(proto string, res *Response) {
			if got, want := res.Header.Get("Got-Length"), fmt.Sprint(wantLen); got != want {
				t.Errorf("Proto %q got length %q; want %q", proto, got, want)
			}
		},
	}.run(t)
}

// Tests that closing the Request.Cancel channel also while still
// reading the response body. Issue 13159.
func TestCancelRequestMidBody_h1(t *testing.T) { testCancelRequestMidBody(t, h1Mode) }
func TestCancelRequestMidBody_h2(t *testing.T) { testCancelRequestMidBody(t, h2Mode) }
func testCancelRequestMidBody(t *testing.T, h2 bool) {
	defer afterTest(t)
	unblock := make(chan bool)
	didFlush := make(chan bool, 1)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		io.WriteString(w, "Hello")
		w.(Flusher).Flush()
		didFlush <- true
		<-unblock
		io.WriteString(w, ", world.")
	}))
	defer cst.close()
	defer close(unblock)

	req, _ := NewRequest("GET", cst.ts.URL, nil)
	cancel := make(chan struct{})
	req.Cancel = cancel

	res, err := cst.c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	<-didFlush

	// Read a bit before we cancel. (Issue 13626)
	// We should have "Hello" at least sitting there.
	firstRead := make([]byte, 10)
	n, err := res.Body.Read(firstRead)
	if err != nil {
		t.Fatal(err)
	}
	firstRead = firstRead[:n]

	close(cancel)

	rest, err := ioutil.ReadAll(res.Body)
	all := string(firstRead) + string(rest)
	if all != "Hello" {
		t.Errorf("Read %q (%q + %q); want Hello", all, firstRead, rest)
	}
	if !reflect.DeepEqual(err, ExportErrRequestCanceled) {
		t.Errorf("ReadAll error = %v; want %v", err, ExportErrRequestCanceled)
	}
}

// Tests that clients can send trailers to a server and that the server can read them.
func TestTrailersClientToServer_h1(t *testing.T) { testTrailersClientToServer(t, h1Mode) }
func TestTrailersClientToServer_h2(t *testing.T) { testTrailersClientToServer(t, h2Mode) }

func testTrailersClientToServer(t *testing.T, h2 bool) {
	defer afterTest(t)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		var decl []string
		for k := range r.Trailer {
			decl = append(decl, k)
		}
		sort.Strings(decl)

		slurp, err := ioutil.ReadAll(r.Body)
		if err != nil {
			t.Errorf("Server reading request body: %v", err)
		}
		if string(slurp) != "foo" {
			t.Errorf("Server read request body %q; want foo", slurp)
		}
		if r.Trailer == nil {
			io.WriteString(w, "nil Trailer")
		} else {
			fmt.Fprintf(w, "decl: %v, vals: %s, %s",
				decl,
				r.Trailer.Get("Client-Trailer-A"),
				r.Trailer.Get("Client-Trailer-B"))
		}
	}))
	defer cst.close()

	var req *Request
	req, _ = NewRequest("POST", cst.ts.URL, io.MultiReader(
		eofReaderFunc(func() {
			req.Trailer["Client-Trailer-A"] = []string{"valuea"}
		}),
		strings.NewReader("foo"),
		eofReaderFunc(func() {
			req.Trailer["Client-Trailer-B"] = []string{"valueb"}
		}),
	))
	req.Trailer = Header{
		"Client-Trailer-A": nil, //  to be set later
		"Client-Trailer-B": nil, //  to be set later
	}
	req.ContentLength = -1
	res, err := cst.c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if err := wantBody(res, err, "decl: [Client-Trailer-A Client-Trailer-B], vals: valuea, valueb"); err != nil {
		t.Error(err)
	}
}

// Tests that servers send trailers to a client and that the client can read them.
func TestTrailersServerToClient_h1(t *testing.T)       { testTrailersServerToClient(t, h1Mode, false) }
func TestTrailersServerToClient_h2(t *testing.T)       { testTrailersServerToClient(t, h2Mode, false) }
func TestTrailersServerToClient_Flush_h1(t *testing.T) { testTrailersServerToClient(t, h1Mode, true) }
func TestTrailersServerToClient_Flush_h2(t *testing.T) { testTrailersServerToClient(t, h2Mode, true) }

func testTrailersServerToClient(t *testing.T, h2, flush bool) {
	defer afterTest(t)
	const body = "Some body"
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Trailer", "Server-Trailer-A, Server-Trailer-B")
		w.Header().Add("Trailer", "Server-Trailer-C")

		io.WriteString(w, body)
		if flush {
			w.(Flusher).Flush()
		}

		// How handlers set Trailers: declare it ahead of time
		// with the Trailer header, and then mutate the
		// Header() of those values later, after the response
		// has been written (we wrote to w above).
		w.Header().Set("Server-Trailer-A", "valuea")
		w.Header().Set("Server-Trailer-C", "valuec") // skipping B
		w.Header().Set("Server-Trailer-NotDeclared", "should be omitted")
	}))
	defer cst.close()

	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}

	wantHeader := Header{
		"Content-Type": {"text/plain; charset=utf-8"},
	}
	wantLen := -1
	if h2 && !flush {
		// In HTTP/1.1, any use of trailers forces HTTP/1.1
		// chunking and a flush at the first write. That's
		// unnecessary with HTTP/2's framing, so the server
		// is able to calculate the length while still sending
		// trailers afterwards.
		wantLen = len(body)
		wantHeader["Content-Length"] = []string{fmt.Sprint(wantLen)}
	}
	if res.ContentLength != int64(wantLen) {
		t.Errorf("ContentLength = %v; want %v", res.ContentLength, wantLen)
	}

	delete(res.Header, "Date") // irrelevant for test
	if !reflect.DeepEqual(res.Header, wantHeader) {
		t.Errorf("Header = %v; want %v", res.Header, wantHeader)
	}

	if got, want := res.Trailer, (Header{
		"Server-Trailer-A": nil,
		"Server-Trailer-B": nil,
		"Server-Trailer-C": nil,
	}); !reflect.DeepEqual(got, want) {
		t.Errorf("Trailer before body read = %v; want %v", got, want)
	}

	if err := wantBody(res, nil, body); err != nil {
		t.Fatal(err)
	}

	if got, want := res.Trailer, (Header{
		"Server-Trailer-A": {"valuea"},
		"Server-Trailer-B": nil,
		"Server-Trailer-C": {"valuec"},
	}); !reflect.DeepEqual(got, want) {
		t.Errorf("Trailer after body read = %v; want %v", got, want)
	}
}

// Don't allow a Body.Read after Body.Close. Issue 13648.
func TestResponseBodyReadAfterClose_h1(t *testing.T) { testResponseBodyReadAfterClose(t, h1Mode) }
func TestResponseBodyReadAfterClose_h2(t *testing.T) { testResponseBodyReadAfterClose(t, h2Mode) }

func testResponseBodyReadAfterClose(t *testing.T, h2 bool) {
	defer afterTest(t)
	const body = "Some body"
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		io.WriteString(w, body)
	}))
	defer cst.close()
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	data, err := ioutil.ReadAll(res.Body)
	if len(data) != 0 || err == nil {
		t.Fatalf("ReadAll returned %q, %v; want error", data, err)
	}
}

func TestConcurrentReadWriteReqBody_h1(t *testing.T) { testConcurrentReadWriteReqBody(t, h1Mode) }
func TestConcurrentReadWriteReqBody_h2(t *testing.T) { testConcurrentReadWriteReqBody(t, h2Mode) }
func testConcurrentReadWriteReqBody(t *testing.T, h2 bool) {
	defer afterTest(t)
	const reqBody = "some request body"
	const resBody = "some response body"
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		var wg sync.WaitGroup
		wg.Add(2)
		didRead := make(chan bool, 1)
		// Read in one goroutine.
		go func() {
			defer wg.Done()
			data, err := ioutil.ReadAll(r.Body)
			if string(data) != reqBody {
				t.Errorf("Handler read %q; want %q", data, reqBody)
			}
			if err != nil {
				t.Errorf("Handler Read: %v", err)
			}
			didRead <- true
		}()
		// Write in another goroutine.
		go func() {
			defer wg.Done()
			if !h2 {
				// our HTTP/1 implementation intentionally
				// doesn't permit writes during read (mostly
				// due to it being undefined); if that is ever
				// relaxed, change this.
				<-didRead
			}
			io.WriteString(w, resBody)
		}()
		wg.Wait()
	}))
	defer cst.close()
	req, _ := NewRequest("POST", cst.ts.URL, strings.NewReader(reqBody))
	req.Header.Add("Expect", "100-continue") // just to complicate things
	res, err := cst.c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	data, err := ioutil.ReadAll(res.Body)
	defer res.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != resBody {
		t.Errorf("read %q; want %q", data, resBody)
	}
}

func TestConnectRequest_h1(t *testing.T) { testConnectRequest(t, h1Mode) }
func TestConnectRequest_h2(t *testing.T) { testConnectRequest(t, h2Mode) }
func testConnectRequest(t *testing.T, h2 bool) {
	defer afterTest(t)
	gotc := make(chan *Request, 1)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		gotc <- r
	}))
	defer cst.close()

	u, err := url.Parse(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		req  *Request
		want string
	}{
		{
			req: &Request{
				Method: "CONNECT",
				Header: Header{},
				URL:    u,
			},
			want: u.Host,
		},
		{
			req: &Request{
				Method: "CONNECT",
				Header: Header{},
				URL:    u,
				Host:   "example.com:123",
			},
			want: "example.com:123",
		},
	}

	for i, tt := range tests {
		res, err := cst.c.Do(tt.req)
		if err != nil {
			t.Errorf("%d. RoundTrip = %v", i, err)
			continue
		}
		res.Body.Close()
		req := <-gotc
		if req.Method != "CONNECT" {
			t.Errorf("method = %q; want CONNECT", req.Method)
		}
		if req.Host != tt.want {
			t.Errorf("Host = %q; want %q", req.Host, tt.want)
		}
		if req.URL.Host != tt.want {
			t.Errorf("URL.Host = %q; want %q", req.URL.Host, tt.want)
		}
	}
}

func TestTransportUserAgent_h1(t *testing.T) { testTransportUserAgent(t, h1Mode) }
func TestTransportUserAgent_h2(t *testing.T) { testTransportUserAgent(t, h2Mode) }
func testTransportUserAgent(t *testing.T, h2 bool) {
	defer afterTest(t)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "%q", r.Header["User-Agent"])
	}))
	defer cst.close()

	either := func(a, b string) string {
		if h2 {
			return b
		}
		return a
	}

	tests := []struct {
		setup func(*Request)
		want  string
	}{
		{
			func(r *Request) {},
			either(`["Go-http-client/1.1"]`, `["Go-http-client/2.0"]`),
		},
		{
			func(r *Request) { r.Header.Set("User-Agent", "foo/1.2.3") },
			`["foo/1.2.3"]`,
		},
		{
			func(r *Request) { r.Header["User-Agent"] = []string{"single", "or", "multiple"} },
			`["single"]`,
		},
		{
			func(r *Request) { r.Header.Set("User-Agent", "") },
			`[]`,
		},
		{
			func(r *Request) { r.Header["User-Agent"] = nil },
			`[]`,
		},
	}
	for i, tt := range tests {
		req, _ := NewRequest("GET", cst.ts.URL, nil)
		tt.setup(req)
		res, err := cst.c.Do(req)
		if err != nil {
			t.Errorf("%d. RoundTrip = %v", i, err)
			continue
		}
		slurp, err := ioutil.ReadAll(res.Body)
		res.Body.Close()
		if err != nil {
			t.Errorf("%d. read body = %v", i, err)
			continue
		}
		if string(slurp) != tt.want {
			t.Errorf("%d. body mismatch.\n got: %s\nwant: %s\n", i, slurp, tt.want)
		}
	}
}

func TestStarRequestFoo_h1(t *testing.T)     { testStarRequest(t, "FOO", h1Mode) }
func TestStarRequestFoo_h2(t *testing.T)     { testStarRequest(t, "FOO", h2Mode) }
func TestStarRequestOptions_h1(t *testing.T) { testStarRequest(t, "OPTIONS", h1Mode) }
func TestStarRequestOptions_h2(t *testing.T) { testStarRequest(t, "OPTIONS", h2Mode) }
func testStarRequest(t *testing.T, method string, h2 bool) {
	defer afterTest(t)
	gotc := make(chan *Request, 1)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("foo", "bar")
		gotc <- r
		w.(Flusher).Flush()
	}))
	defer cst.close()

	u, err := url.Parse(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	u.Path = "*"

	req := &Request{
		Method: method,
		Header: Header{},
		URL:    u,
	}

	res, err := cst.c.Do(req)
	if err != nil {
		t.Fatalf("RoundTrip = %v", err)
	}
	res.Body.Close()

	wantFoo := "bar"
	wantLen := int64(-1)
	if method == "OPTIONS" {
		wantFoo = ""
		wantLen = 0
	}
	if res.StatusCode != 200 {
		t.Errorf("status code = %v; want %d", res.Status, 200)
	}
	if res.ContentLength != wantLen {
		t.Errorf("content length = %v; want %d", res.ContentLength, wantLen)
	}
	if got := res.Header.Get("foo"); got != wantFoo {
		t.Errorf("response \"foo\" header = %q; want %q", got, wantFoo)
	}
	select {
	case req = <-gotc:
	default:
		req = nil
	}
	if req == nil {
		if method != "OPTIONS" {
			t.Fatalf("handler never got request")
		}
		return
	}
	if req.Method != method {
		t.Errorf("method = %q; want %q", req.Method, method)
	}
	if req.URL.Path != "*" {
		t.Errorf("URL.Path = %q; want *", req.URL.Path)
	}
	if req.RequestURI != "*" {
		t.Errorf("RequestURI = %q; want *", req.RequestURI)
	}
}

// Issue 13957
func TestTransportDiscardsUnneededConns(t *testing.T) {
	setParallel(t)
	defer afterTest(t)
	cst := newClientServerTest(t, h2Mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "Hello, %v", r.RemoteAddr)
	}))
	defer cst.close()

	var numOpen, numClose int32 // atomic

	tlsConfig := &tls.Config{InsecureSkipVerify: true}
	tr := &Transport{
		TLSClientConfig: tlsConfig,
		DialTLS: func(_, addr string) (net.Conn, error) {
			time.Sleep(10 * time.Millisecond)
			rc, err := net.Dial("tcp", addr)
			if err != nil {
				return nil, err
			}
			atomic.AddInt32(&numOpen, 1)
			c := noteCloseConn{rc, func() { atomic.AddInt32(&numClose, 1) }}
			return tls.Client(c, tlsConfig), nil
		},
	}
	if err := ExportHttp2ConfigureTransport(tr); err != nil {
		t.Fatal(err)
	}
	defer tr.CloseIdleConnections()

	c := &Client{Transport: tr}

	const N = 10
	gotBody := make(chan string, N)
	var wg sync.WaitGroup
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := c.Get(cst.ts.URL)
			if err != nil {
				t.Errorf("Get: %v", err)
				return
			}
			defer resp.Body.Close()
			slurp, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				t.Error(err)
			}
			gotBody <- string(slurp)
		}()
	}
	wg.Wait()
	close(gotBody)

	var last string
	for got := range gotBody {
		if last == "" {
			last = got
			continue
		}
		if got != last {
			t.Errorf("Response body changed: %q -> %q", last, got)
		}
	}

	var open, close int32
	for i := 0; i < 150; i++ {
		open, close = atomic.LoadInt32(&numOpen), atomic.LoadInt32(&numClose)
		if open < 1 {
			t.Fatalf("open = %d; want at least", open)
		}
		if close == open-1 {
			// Success
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Errorf("%d connections opened, %d closed; want %d to close", open, close, open-1)
}

// tests that Transport doesn't retain a pointer to the provided request.
func TestTransportGCRequest_Body_h1(t *testing.T)   { testTransportGCRequest(t, h1Mode, true) }
func TestTransportGCRequest_Body_h2(t *testing.T)   { testTransportGCRequest(t, h2Mode, true) }
func TestTransportGCRequest_NoBody_h1(t *testing.T) { testTransportGCRequest(t, h1Mode, false) }
func TestTransportGCRequest_NoBody_h2(t *testing.T) { testTransportGCRequest(t, h2Mode, false) }
func testTransportGCRequest(t *testing.T, h2, body bool) {
	setParallel(t)
	defer afterTest(t)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		ioutil.ReadAll(r.Body)
		if body {
			io.WriteString(w, "Hello.")
		}
	}))
	defer cst.close()

	didGC := make(chan struct{})
	(func() {
		body := strings.NewReader("some body")
		req, _ := NewRequest("POST", cst.ts.URL, body)
		runtime.SetFinalizer(req, func(*Request) { close(didGC) })
		res, err := cst.c.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := ioutil.ReadAll(res.Body); err != nil {
			t.Fatal(err)
		}
		if err := res.Body.Close(); err != nil {
			t.Fatal(err)
		}
	})()
	timeout := time.NewTimer(5 * time.Second)
	defer timeout.Stop()
	for {
		select {
		case <-didGC:
			return
		case <-time.After(100 * time.Millisecond):
			runtime.GC()
		case <-timeout.C:
			t.Fatal("never saw GC of request")
		}
	}
}

func TestTransportRejectsInvalidHeaders_h1(t *testing.T) {
	testTransportRejectsInvalidHeaders(t, h1Mode)
}
func TestTransportRejectsInvalidHeaders_h2(t *testing.T) {
	testTransportRejectsInvalidHeaders(t, h2Mode)
}
func testTransportRejectsInvalidHeaders(t *testing.T, h2 bool) {
	setParallel(t)
	defer afterTest(t)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "Handler saw headers: %q", r.Header)
	}), optQuietLog)
	defer cst.close()
	cst.tr.DisableKeepAlives = true

	tests := []struct {
		key, val string
		ok       bool
	}{
		{"Foo", "capital-key", true}, // verify h2 allows capital keys
		{"Foo", "foo\x00bar", false}, // \x00 byte in value not allowed
		{"Foo", "two\nlines", false}, // \n byte in value not allowed
		{"bogus\nkey", "v", false},   // \n byte also not allowed in key
		{"A space", "v", false},      // spaces in keys not allowed
		{"имя", "v", false},          // key must be ascii
		{"name", "валю", true},       // value may be non-ascii
		{"", "v", false},             // key must be non-empty
		{"k", "", true},              // value may be empty
	}
	for _, tt := range tests {
		dialedc := make(chan bool, 1)
		cst.tr.Dial = func(netw, addr string) (net.Conn, error) {
			dialedc <- true
			return net.Dial(netw, addr)
		}
		req, _ := NewRequest("GET", cst.ts.URL, nil)
		req.Header[tt.key] = []string{tt.val}
		res, err := cst.c.Do(req)
		var body []byte
		if err == nil {
			body, _ = ioutil.ReadAll(res.Body)
			res.Body.Close()
		}
		var dialed bool
		select {
		case <-dialedc:
			dialed = true
		default:
		}

		if !tt.ok && dialed {
			t.Errorf("For key %q, value %q, transport dialed. Expected local failure. Response was: (%v, %v)\nServer replied with: %s", tt.key, tt.val, res, err, body)
		} else if (err == nil) != tt.ok {
			t.Errorf("For key %q, value %q; got err = %v; want ok=%v", tt.key, tt.val, err, tt.ok)
		}
	}
}

// Tests that we support bogus under-100 HTTP statuses, because we historically
// have. This might change at some point, but not yet in Go 1.6.
func TestBogusStatusWorks_h1(t *testing.T) { testBogusStatusWorks(t, h1Mode) }
func TestBogusStatusWorks_h2(t *testing.T) { testBogusStatusWorks(t, h2Mode) }
func testBogusStatusWorks(t *testing.T, h2 bool) {
	defer afterTest(t)
	const code = 7
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.WriteHeader(code)
	}))
	defer cst.close()

	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != code {
		t.Errorf("StatusCode = %d; want %d", res.StatusCode, code)
	}
}

func TestInterruptWithPanic_h1(t *testing.T)     { testInterruptWithPanic(t, h1Mode, "boom") }
func TestInterruptWithPanic_h2(t *testing.T)     { testInterruptWithPanic(t, h2Mode, "boom") }
func TestInterruptWithPanic_nil_h1(t *testing.T) { testInterruptWithPanic(t, h1Mode, nil) }
func TestInterruptWithPanic_nil_h2(t *testing.T) { testInterruptWithPanic(t, h2Mode, nil) }
func TestInterruptWithPanic_ErrAbortHandler_h1(t *testing.T) {
	testInterruptWithPanic(t, h1Mode, ErrAbortHandler)
}
func TestInterruptWithPanic_ErrAbortHandler_h2(t *testing.T) {
	testInterruptWithPanic(t, h2Mode, ErrAbortHandler)
}
func testInterruptWithPanic(t *testing.T, h2 bool, panicValue interface{}) {
	setParallel(t)
	const msg = "hello"
	defer afterTest(t)

	testDone := make(chan struct{})
	defer close(testDone)

	var errorLog lockedBytesBuffer
	gotHeaders := make(chan bool, 1)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		io.WriteString(w, msg)
		w.(Flusher).Flush()

		select {
		case <-gotHeaders:
		case <-testDone:
		}
		panic(panicValue)
	}), func(ts *httptest.Server) {
		ts.Config.ErrorLog = log.New(&errorLog, "", 0)
	})
	defer cst.close()
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	gotHeaders <- true
	defer res.Body.Close()
	slurp, err := ioutil.ReadAll(res.Body)
	if string(slurp) != msg {
		t.Errorf("client read %q; want %q", slurp, msg)
	}
	if err == nil {
		t.Errorf("client read all successfully; want some error")
	}
	logOutput := func() string {
		errorLog.Lock()
		defer errorLog.Unlock()
		return errorLog.String()
	}
	wantStackLogged := panicValue != nil && panicValue != ErrAbortHandler

	if err := waitErrCondition(5*time.Second, 10*time.Millisecond, func() error {
		gotLog := logOutput()
		if !wantStackLogged {
			if gotLog == "" {
				return nil
			}
			return fmt.Errorf("want no log output; got: %s", gotLog)
		}
		if gotLog == "" {
			return fmt.Errorf("wanted a stack trace logged; got nothing")
		}
		if !strings.Contains(gotLog, "created by ") && strings.Count(gotLog, "\n") < 6 {
			return fmt.Errorf("output doesn't look like a panic stack trace. Got: %s", gotLog)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

type lockedBytesBuffer struct {
	sync.Mutex
	bytes.Buffer
}

func (b *lockedBytesBuffer) Write(p []byte) (int, error) {
	b.Lock()
	defer b.Unlock()
	return b.Buffer.Write(p)
}

// Issue 15366
func TestH12_AutoGzipWithDumpResponse(t *testing.T) {
	h12Compare{
		Handler: func(w ResponseWriter, r *Request) {
			h := w.Header()
			h.Set("Content-Encoding", "gzip")
			h.Set("Content-Length", "23")
			h.Set("Connection", "keep-alive")
			io.WriteString(w, "\x1f\x8b\b\x00\x00\x00\x00\x00\x00\x00s\xf3\xf7\a\x00\xab'\xd4\x1a\x03\x00\x00\x00")
		},
		EarlyCheckResponse: func(proto string, res *Response) {
			if !res.Uncompressed {
				t.Errorf("%s: expected Uncompressed to be set", proto)
			}
			dump, err := httputil.DumpResponse(res, true)
			if err != nil {
				t.Errorf("%s: DumpResponse: %v", proto, err)
				return
			}
			if strings.Contains(string(dump), "Connection: close") {
				t.Errorf("%s: should not see \"Connection: close\" in dump; got:\n%s", proto, dump)
			}
			if !strings.Contains(string(dump), "FOO") {
				t.Errorf("%s: should see \"FOO\" in response; got:\n%s", proto, dump)
			}
		},
	}.run(t)
}

// Issue 14607
func TestCloseIdleConnections_h1(t *testing.T) { testCloseIdleConnections(t, h1Mode) }
func TestCloseIdleConnections_h2(t *testing.T) { testCloseIdleConnections(t, h2Mode) }
func testCloseIdleConnections(t *testing.T, h2 bool) {
	setParallel(t)
	defer afterTest(t)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("X-Addr", r.RemoteAddr)
	}))
	defer cst.close()
	get := func() string {
		res, err := cst.c.Get(cst.ts.URL)
		if err != nil {
			t.Fatal(err)
		}
		res.Body.Close()
		v := res.Header.Get("X-Addr")
		if v == "" {
			t.Fatal("didn't get X-Addr")
		}
		return v
	}
	a1 := get()
	cst.tr.CloseIdleConnections()
	a2 := get()
	if a1 == a2 {
		t.Errorf("didn't close connection")
	}
}

type noteCloseConn struct {
	net.Conn
	closeFunc func()
}

func (x noteCloseConn) Close() error {
	x.closeFunc()
	return x.Conn.Close()
}

type testErrorReader struct{ t *testing.T }

func (r testErrorReader) Read(p []byte) (n int, err error) {
	r.t.Error("unexpected Read call")
	return 0, io.EOF
}

func TestNoSniffExpectRequestBody_h1(t *testing.T) { testNoSniffExpectRequestBody(t, h1Mode) }
func TestNoSniffExpectRequestBody_h2(t *testing.T) { testNoSniffExpectRequestBody(t, h2Mode) }

func testNoSniffExpectRequestBody(t *testing.T, h2 bool) {
	defer afterTest(t)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.WriteHeader(StatusUnauthorized)
	}))
	defer cst.close()

	// Set ExpectContinueTimeout non-zero so RoundTrip won't try to write it.
	cst.tr.ExpectContinueTimeout = 10 * time.Second

	req, err := NewRequest("POST", cst.ts.URL, testErrorReader{t})
	if err != nil {
		t.Fatal(err)
	}
	req.ContentLength = 0 // so transport is tempted to sniff it
	req.Header.Set("Expect", "100-continue")
	res, err := cst.tr.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != StatusUnauthorized {
		t.Errorf("status code = %v; want %v", res.StatusCode, StatusUnauthorized)
	}
}

func TestServerUndeclaredTrailers_h1(t *testing.T) { testServerUndeclaredTrailers(t, h1Mode) }
func TestServerUndeclaredTrailers_h2(t *testing.T) { testServerUndeclaredTrailers(t, h2Mode) }
func testServerUndeclaredTrailers(t *testing.T, h2 bool) {
	defer afterTest(t)
	cst := newClientServerTest(t, h2, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Foo", "Bar")
		w.Header().Set("Trailer:Foo", "Baz")
		w.(Flusher).Flush()
		w.Header().Add("Trailer:Foo", "Baz2")
		w.Header().Set("Trailer:Bar", "Quux")
	}))
	defer cst.close()
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.Copy(ioutil.Discard, res.Body); err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	delete(res.Header, "Date")
	delete(res.Header, "Content-Type")

	if want := (Header{"Foo": {"Bar"}}); !reflect.DeepEqual(res.Header, want) {
		t.Errorf("Header = %#v; want %#v", res.Header, want)
	}
	if want := (Header{"Foo": {"Baz", "Baz2"}, "Bar": {"Quux"}}); !reflect.DeepEqual(res.Trailer, want) {
		t.Errorf("Trailer = %#v; want %#v", res.Trailer, want)
	}
}
