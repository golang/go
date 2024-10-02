// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests that use both the client & server, in both HTTP/1 and HTTP/2 mode.

package http_test

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/rand"
	"crypto/sha1"
	"crypto/tls"
	"fmt"
	"hash"
	"io"
	"log"
	"maps"
	"net"
	. "net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"net/http/httputil"
	"net/textproto"
	"net/url"
	"os"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type testMode string

const (
	http1Mode  = testMode("h1")     // HTTP/1.1
	https1Mode = testMode("https1") // HTTPS/1.1
	http2Mode  = testMode("h2")     // HTTP/2
)

type testNotParallelOpt struct{}

var (
	testNotParallel = testNotParallelOpt{}
)

type TBRun[T any] interface {
	testing.TB
	Run(string, func(T)) bool
}

// run runs a client/server test in a variety of test configurations.
//
// Tests execute in HTTP/1.1 and HTTP/2 modes by default.
// To run in a different set of configurations, pass a []testMode option.
//
// Tests call t.Parallel() by default.
// To disable parallel execution, pass the testNotParallel option.
func run[T TBRun[T]](t T, f func(t T, mode testMode), opts ...any) {
	t.Helper()
	modes := []testMode{http1Mode, http2Mode}
	parallel := true
	for _, opt := range opts {
		switch opt := opt.(type) {
		case []testMode:
			modes = opt
		case testNotParallelOpt:
			parallel = false
		default:
			t.Fatalf("unknown option type %T", opt)
		}
	}
	if t, ok := any(t).(*testing.T); ok && parallel {
		setParallel(t)
	}
	for _, mode := range modes {
		t.Run(string(mode), func(t T) {
			t.Helper()
			if t, ok := any(t).(*testing.T); ok && parallel {
				setParallel(t)
			}
			t.Cleanup(func() {
				afterTest(t)
			})
			f(t, mode)
		})
	}
}

type clientServerTest struct {
	t  testing.TB
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
	slurp, err := io.ReadAll(res.Body)
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

var optQuietLog = func(ts *httptest.Server) {
	ts.Config.ErrorLog = quietLog
}

func optWithServerLog(lg *log.Logger) func(*httptest.Server) {
	return func(ts *httptest.Server) {
		ts.Config.ErrorLog = lg
	}
}

// newClientServerTest creates and starts an httptest.Server.
//
// The mode parameter selects the implementation to test:
// HTTP/1, HTTP/2, etc. Tests using newClientServerTest should use
// the 'run' function, which will start a subtests for each tested mode.
//
// The vararg opts parameter can include functions to configure the
// test server or transport.
//
//	func(*httptest.Server) // run before starting the server
//	func(*http.Transport)
func newClientServerTest(t testing.TB, mode testMode, h Handler, opts ...any) *clientServerTest {
	if mode == http2Mode {
		CondSkipHTTP2(t)
	}
	cst := &clientServerTest{
		t:  t,
		h2: mode == http2Mode,
		h:  h,
	}
	cst.ts = httptest.NewUnstartedServer(h)

	var transportFuncs []func(*Transport)
	for _, opt := range opts {
		switch opt := opt.(type) {
		case func(*Transport):
			transportFuncs = append(transportFuncs, opt)
		case func(*httptest.Server):
			opt(cst.ts)
		default:
			t.Fatalf("unhandled option type %T", opt)
		}
	}

	if cst.ts.Config.ErrorLog == nil {
		cst.ts.Config.ErrorLog = log.New(testLogWriter{t}, "", 0)
	}

	switch mode {
	case http1Mode:
		cst.ts.Start()
	case https1Mode:
		cst.ts.StartTLS()
	case http2Mode:
		ExportHttp2ConfigureServer(cst.ts.Config, nil)
		cst.ts.TLS = cst.ts.Config.TLSConfig
		cst.ts.StartTLS()
	default:
		t.Fatalf("unknown test mode %v", mode)
	}
	cst.c = cst.ts.Client()
	cst.tr = cst.c.Transport.(*Transport)
	if mode == http2Mode {
		if err := ExportHttp2ConfigureTransport(cst.tr); err != nil {
			t.Fatal(err)
		}
	}
	for _, f := range transportFuncs {
		f(cst.tr)
	}
	t.Cleanup(func() {
		cst.close()
	})
	return cst
}

type testLogWriter struct {
	t testing.TB
}

func (w testLogWriter) Write(b []byte) (int, error) {
	w.t.Logf("server log: %v", strings.TrimSpace(string(b)))
	return len(b), nil
}

// Testing the newClientServerTest helper itself.
func TestNewClientServerTest(t *testing.T) {
	run(t, testNewClientServerTest, []testMode{http1Mode, https1Mode, http2Mode})
}
func testNewClientServerTest(t *testing.T, mode testMode) {
	var got struct {
		sync.Mutex
		proto  string
		hasTLS bool
	}
	h := HandlerFunc(func(w ResponseWriter, r *Request) {
		got.Lock()
		defer got.Unlock()
		got.proto = r.Proto
		got.hasTLS = r.TLS != nil
	})
	cst := newClientServerTest(t, mode, h)
	if _, err := cst.c.Head(cst.ts.URL); err != nil {
		t.Fatal(err)
	}
	var wantProto string
	var wantTLS bool
	switch mode {
	case http1Mode:
		wantProto = "HTTP/1.1"
		wantTLS = false
	case https1Mode:
		wantProto = "HTTP/1.1"
		wantTLS = true
	case http2Mode:
		wantProto = "HTTP/2.0"
		wantTLS = true
	}
	if got.proto != wantProto {
		t.Errorf("req.Proto = %q, want %q", got.proto, wantProto)
	}
	if got.hasTLS != wantTLS {
		t.Errorf("req.TLS set: %v, want %v", got.hasTLS, wantTLS)
	}
}

func TestChunkedResponseHeaders(t *testing.T) { run(t, testChunkedResponseHeaders) }
func testChunkedResponseHeaders(t *testing.T, mode testMode) {
	log.SetOutput(io.Discard) // is noisy otherwise
	defer log.SetOutput(os.Stderr)
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Content-Length", "intentional gibberish") // we check that this is deleted
		w.(Flusher).Flush()
		fmt.Fprintf(w, "I am a chunked response.")
	}))

	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatalf("Get error: %v", err)
	}
	defer res.Body.Close()
	if g, e := res.ContentLength, int64(-1); g != e {
		t.Errorf("expected ContentLength of %d; got %d", e, g)
	}
	wantTE := []string{"chunked"}
	if mode == http2Mode {
		wantTE = nil
	}
	if !slices.Equal(res.TransferEncoding, wantTE) {
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
	Opts               []any
}

func (tt h12Compare) reqFunc() reqFunc {
	if tt.ReqFunc == nil {
		return (*Client).Get
	}
	return tt.ReqFunc
}

func (tt h12Compare) run(t *testing.T) {
	setParallel(t)
	cst1 := newClientServerTest(t, http1Mode, HandlerFunc(tt.Handler), tt.Opts...)
	defer cst1.close()
	cst2 := newClientServerTest(t, http2Mode, HandlerFunc(tt.Handler), tt.Opts...)
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
	if res.Proto == wantProto || res.Proto == "HTTP/IGNORE" {
		res.Proto, res.ProtoMajor, res.ProtoMinor = "", 0, 0
	} else {
		t.Errorf("got %q response; want %q", res.Proto, wantProto)
	}
	slurp, err := io.ReadAll(res.Body)

	res.Body.Close()
	res.Body = slurpResult{
		ReadCloser: io.NopCloser(bytes.NewReader(slurp)),
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
func TestHandlerWritesTooMuch(t *testing.T) { run(t, testHandlerWritesTooMuch) }
func testHandlerWritesTooMuch(t *testing.T, mode testMode) {
	wantBody := []byte("123")
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		rc := NewResponseController(w)
		w.Header().Set("Content-Length", fmt.Sprintf("%v", len(wantBody)))
		rc.Flush()
		w.Write(wantBody)
		rc.Flush()
		n, err := io.WriteString(w, "x") // too many
		if err == nil {
			err = rc.Flush()
		}
		// TODO: Check that this is ErrContentLength, not just any error.
		if err == nil {
			t.Errorf("for proto %q, final write = %v, %v; want _, some error", r.Proto, n, err)
		}
	}))

	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()

	gotBody, _ := io.ReadAll(res.Body)
	if !bytes.Equal(gotBody, wantBody) {
		t.Fatalf("got response body: %q; want %q", gotBody, wantBody)
	}
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
		Opts: []any{
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
func Test304Responses(t *testing.T) { run(t, test304Responses) }
func test304Responses(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
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
	body, err := io.ReadAll(res.Body)
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
func TestCancelRequestMidBody(t *testing.T) { run(t, testCancelRequestMidBody) }
func testCancelRequestMidBody(t *testing.T, mode testMode) {
	unblock := make(chan bool)
	didFlush := make(chan bool, 1)
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		io.WriteString(w, "Hello")
		w.(Flusher).Flush()
		didFlush <- true
		<-unblock
		io.WriteString(w, ", world.")
	}))
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

	rest, err := io.ReadAll(res.Body)
	all := string(firstRead) + string(rest)
	if all != "Hello" {
		t.Errorf("Read %q (%q + %q); want Hello", all, firstRead, rest)
	}
	if err != ExportErrRequestCanceled {
		t.Errorf("ReadAll error = %v; want %v", err, ExportErrRequestCanceled)
	}
}

// Tests that clients can send trailers to a server and that the server can read them.
func TestTrailersClientToServer(t *testing.T) { run(t, testTrailersClientToServer) }
func testTrailersClientToServer(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		slurp, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("Server reading request body: %v", err)
		}
		if string(slurp) != "foo" {
			t.Errorf("Server read request body %q; want foo", slurp)
		}
		if r.Trailer == nil {
			io.WriteString(w, "nil Trailer")
		} else {
			decl := slices.Sorted(maps.Keys(r.Trailer))
			fmt.Fprintf(w, "decl: %v, vals: %s, %s",
				decl,
				r.Trailer.Get("Client-Trailer-A"),
				r.Trailer.Get("Client-Trailer-B"))
		}
	}))

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
func TestTrailersServerToClient(t *testing.T) {
	run(t, func(t *testing.T, mode testMode) {
		testTrailersServerToClient(t, mode, false)
	})
}
func TestTrailersServerToClientFlush(t *testing.T) {
	run(t, func(t *testing.T, mode testMode) {
		testTrailersServerToClient(t, mode, true)
	})
}

func testTrailersServerToClient(t *testing.T, mode testMode, flush bool) {
	const body = "Some body"
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
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

	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}

	wantHeader := Header{
		"Content-Type": {"text/plain; charset=utf-8"},
	}
	wantLen := -1
	if mode == http2Mode && !flush {
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
func TestResponseBodyReadAfterClose(t *testing.T) { run(t, testResponseBodyReadAfterClose) }
func testResponseBodyReadAfterClose(t *testing.T, mode testMode) {
	const body = "Some body"
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		io.WriteString(w, body)
	}))
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	data, err := io.ReadAll(res.Body)
	if len(data) != 0 || err == nil {
		t.Fatalf("ReadAll returned %q, %v; want error", data, err)
	}
}

func TestConcurrentReadWriteReqBody(t *testing.T) { run(t, testConcurrentReadWriteReqBody) }
func testConcurrentReadWriteReqBody(t *testing.T, mode testMode) {
	const reqBody = "some request body"
	const resBody = "some response body"
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		var wg sync.WaitGroup
		wg.Add(2)
		didRead := make(chan bool, 1)
		// Read in one goroutine.
		go func() {
			defer wg.Done()
			data, err := io.ReadAll(r.Body)
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
			if mode != http2Mode {
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
	req, _ := NewRequest("POST", cst.ts.URL, strings.NewReader(reqBody))
	req.Header.Add("Expect", "100-continue") // just to complicate things
	res, err := cst.c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	data, err := io.ReadAll(res.Body)
	defer res.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != resBody {
		t.Errorf("read %q; want %q", data, resBody)
	}
}

func TestConnectRequest(t *testing.T) { run(t, testConnectRequest) }
func testConnectRequest(t *testing.T, mode testMode) {
	gotc := make(chan *Request, 1)
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		gotc <- r
	}))

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

func TestTransportUserAgent(t *testing.T) { run(t, testTransportUserAgent) }
func testTransportUserAgent(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "%q", r.Header["User-Agent"])
	}))

	either := func(a, b string) string {
		if mode == http2Mode {
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
		slurp, err := io.ReadAll(res.Body)
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

func TestStarRequestMethod(t *testing.T) {
	for _, method := range []string{"FOO", "OPTIONS"} {
		t.Run(method, func(t *testing.T) {
			run(t, func(t *testing.T, mode testMode) {
				testStarRequest(t, method, mode)
			})
		})
	}
}
func testStarRequest(t *testing.T, method string, mode testMode) {
	gotc := make(chan *Request, 1)
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("foo", "bar")
		gotc <- r
		w.(Flusher).Flush()
	}))

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
	run(t, testTransportDiscardsUnneededConns, []testMode{http2Mode})
}
func testTransportDiscardsUnneededConns(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
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
				// Try to work around spurious connection reset on loaded system.
				// See golang.org/issue/33585 and golang.org/issue/36797.
				time.Sleep(10 * time.Millisecond)
				resp, err = c.Get(cst.ts.URL)
				if err != nil {
					t.Errorf("Get: %v", err)
					return
				}
			}
			defer resp.Body.Close()
			slurp, err := io.ReadAll(resp.Body)
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
func TestTransportGCRequest(t *testing.T) {
	run(t, func(t *testing.T, mode testMode) {
		t.Run("Body", func(t *testing.T) { testTransportGCRequest(t, mode, true) })
		t.Run("NoBody", func(t *testing.T) { testTransportGCRequest(t, mode, false) })
	})
}
func testTransportGCRequest(t *testing.T, mode testMode, body bool) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		io.ReadAll(r.Body)
		if body {
			io.WriteString(w, "Hello.")
		}
	}))

	didGC := make(chan struct{})
	(func() {
		body := strings.NewReader("some body")
		req, _ := NewRequest("POST", cst.ts.URL, body)
		runtime.SetFinalizer(req, func(*Request) { close(didGC) })
		res, err := cst.c.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := io.ReadAll(res.Body); err != nil {
			t.Fatal(err)
		}
		if err := res.Body.Close(); err != nil {
			t.Fatal(err)
		}
	})()
	for {
		select {
		case <-didGC:
			return
		case <-time.After(1 * time.Millisecond):
			runtime.GC()
		}
	}
}

func TestTransportRejectsInvalidHeaders(t *testing.T) { run(t, testTransportRejectsInvalidHeaders) }
func testTransportRejectsInvalidHeaders(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "Handler saw headers: %q", r.Header)
	}), optQuietLog)
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
			body, _ = io.ReadAll(res.Body)
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

func TestInterruptWithPanic(t *testing.T) {
	run(t, func(t *testing.T, mode testMode) {
		t.Run("boom", func(t *testing.T) { testInterruptWithPanic(t, mode, "boom") })
		t.Run("nil", func(t *testing.T) { t.Setenv("GODEBUG", "panicnil=1"); testInterruptWithPanic(t, mode, nil) })
		t.Run("ErrAbortHandler", func(t *testing.T) { testInterruptWithPanic(t, mode, ErrAbortHandler) })
	}, testNotParallel)
}
func testInterruptWithPanic(t *testing.T, mode testMode, panicValue any) {
	const msg = "hello"

	testDone := make(chan struct{})
	defer close(testDone)

	var errorLog lockedBytesBuffer
	gotHeaders := make(chan bool, 1)
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
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
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	gotHeaders <- true
	defer res.Body.Close()
	slurp, err := io.ReadAll(res.Body)
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

	waitCondition(t, 10*time.Millisecond, func(d time.Duration) bool {
		gotLog := logOutput()
		if !wantStackLogged {
			if gotLog == "" {
				return true
			}
			t.Fatalf("want no log output; got: %s", gotLog)
		}
		if gotLog == "" {
			if d > 0 {
				t.Logf("wanted a stack trace logged; got nothing after %v", d)
			}
			return false
		}
		if !strings.Contains(gotLog, "created by ") && strings.Count(gotLog, "\n") < 6 {
			if d > 0 {
				t.Logf("output doesn't look like a panic stack trace after %v. Got: %s", d, gotLog)
			}
			return false
		}
		return true
	})
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
func TestCloseIdleConnections(t *testing.T) { run(t, testCloseIdleConnections) }
func testCloseIdleConnections(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("X-Addr", r.RemoteAddr)
	}))
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

func TestNoSniffExpectRequestBody(t *testing.T) { run(t, testNoSniffExpectRequestBody) }
func testNoSniffExpectRequestBody(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.WriteHeader(StatusUnauthorized)
	}))

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

func TestServerUndeclaredTrailers(t *testing.T) { run(t, testServerUndeclaredTrailers) }
func testServerUndeclaredTrailers(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		w.Header().Set("Foo", "Bar")
		w.Header().Set("Trailer:Foo", "Baz")
		w.(Flusher).Flush()
		w.Header().Add("Trailer:Foo", "Baz2")
		w.Header().Set("Trailer:Bar", "Quux")
	}))
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.Copy(io.Discard, res.Body); err != nil {
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

func TestBadResponseAfterReadingBody(t *testing.T) {
	run(t, testBadResponseAfterReadingBody, []testMode{http1Mode})
}
func testBadResponseAfterReadingBody(t *testing.T, mode testMode) {
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		_, err := io.Copy(io.Discard, r.Body)
		if err != nil {
			t.Fatal(err)
		}
		c, _, err := w.(Hijacker).Hijack()
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		fmt.Fprintln(c, "some bogus crap")
	}))

	closes := 0
	res, err := cst.c.Post(cst.ts.URL, "text/plain", countCloseReader{&closes, strings.NewReader("hello")})
	if err == nil {
		res.Body.Close()
		t.Fatal("expected an error to be returned from Post")
	}
	if closes != 1 {
		t.Errorf("closes = %d; want 1", closes)
	}
}

func TestWriteHeader0(t *testing.T) { run(t, testWriteHeader0) }
func testWriteHeader0(t *testing.T, mode testMode) {
	gotpanic := make(chan bool, 1)
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		defer close(gotpanic)
		defer func() {
			if e := recover(); e != nil {
				got := fmt.Sprintf("%T, %v", e, e)
				want := "string, invalid WriteHeader code 0"
				if got != want {
					t.Errorf("unexpected panic value:\n got: %v\nwant: %v\n", got, want)
				}
				gotpanic <- true

				// Set an explicit 503. This also tests that the WriteHeader call panics
				// before it recorded that an explicit value was set and that bogus
				// value wasn't stuck.
				w.WriteHeader(503)
			}
		}()
		w.WriteHeader(0)
	}))
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != 503 {
		t.Errorf("Response: %v %q; want 503", res.StatusCode, res.Status)
	}
	if !<-gotpanic {
		t.Error("expected panic in handler")
	}
}

// Issue 23010: don't be super strict checking WriteHeader's code if
// it's not even valid to call WriteHeader then anyway.
func TestWriteHeaderNoCodeCheck(t *testing.T) {
	run(t, func(t *testing.T, mode testMode) {
		testWriteHeaderAfterWrite(t, mode, false)
	})
}
func TestWriteHeaderNoCodeCheck_h1hijack(t *testing.T) {
	testWriteHeaderAfterWrite(t, http1Mode, true)
}
func testWriteHeaderAfterWrite(t *testing.T, mode testMode, hijack bool) {
	var errorLog lockedBytesBuffer
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		if hijack {
			conn, _, _ := w.(Hijacker).Hijack()
			defer conn.Close()
			conn.Write([]byte("HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\nfoo"))
			w.WriteHeader(0) // verify this doesn't panic if there's already output; Issue 23010
			conn.Write([]byte("bar"))
			return
		}
		io.WriteString(w, "foo")
		w.(Flusher).Flush()
		w.WriteHeader(0) // verify this doesn't panic if there's already output; Issue 23010
		io.WriteString(w, "bar")
	}), func(ts *httptest.Server) {
		ts.Config.ErrorLog = log.New(&errorLog, "", 0)
	})
	res, err := cst.c.Get(cst.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	body, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(body), "foobar"; got != want {
		t.Errorf("got = %q; want %q", got, want)
	}

	// Also check the stderr output:
	if mode == http2Mode {
		// TODO: also emit this log message for HTTP/2?
		// We historically haven't, so don't check.
		return
	}
	gotLog := strings.TrimSpace(errorLog.String())
	wantLog := "http: superfluous response.WriteHeader call from net/http_test.testWriteHeaderAfterWrite.func1 (clientserver_test.go:"
	if hijack {
		wantLog = "http: response.WriteHeader on hijacked connection from net/http_test.testWriteHeaderAfterWrite.func1 (clientserver_test.go:"
	}
	if !strings.HasPrefix(gotLog, wantLog) {
		t.Errorf("stderr output = %q; want %q", gotLog, wantLog)
	}
}

func TestBidiStreamReverseProxy(t *testing.T) {
	run(t, testBidiStreamReverseProxy, []testMode{http2Mode})
}
func testBidiStreamReverseProxy(t *testing.T, mode testMode) {
	backend := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		if _, err := io.Copy(w, r.Body); err != nil {
			log.Printf("bidi backend copy: %v", err)
		}
	}))

	backURL, err := url.Parse(backend.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	rp := httputil.NewSingleHostReverseProxy(backURL)
	rp.Transport = backend.tr
	proxy := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		rp.ServeHTTP(w, r)
	}))

	bodyRes := make(chan any, 1) // error or hash.Hash
	pr, pw := io.Pipe()
	req, _ := NewRequest("PUT", proxy.ts.URL, pr)
	const size = 4 << 20
	go func() {
		h := sha1.New()
		_, err := io.CopyN(io.MultiWriter(h, pw), rand.Reader, size)
		go pw.Close()
		if err != nil {
			bodyRes <- err
		} else {
			bodyRes <- h
		}
	}()
	res, err := backend.c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	hgot := sha1.New()
	n, err := io.Copy(hgot, res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if n != size {
		t.Fatalf("got %d bytes; want %d", n, size)
	}
	select {
	case v := <-bodyRes:
		switch v := v.(type) {
		default:
			t.Fatalf("body copy: %v", err)
		case hash.Hash:
			if !bytes.Equal(v.Sum(nil), hgot.Sum(nil)) {
				t.Errorf("written bytes didn't match received bytes")
			}
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timeout")
	}

}

// Always use HTTP/1.1 for WebSocket upgrades.
func TestH12_WebSocketUpgrade(t *testing.T) {
	h12Compare{
		Handler: func(w ResponseWriter, r *Request) {
			h := w.Header()
			h.Set("Foo", "bar")
		},
		ReqFunc: func(c *Client, url string) (*Response, error) {
			req, _ := NewRequest("GET", url, nil)
			req.Header.Set("Connection", "Upgrade")
			req.Header.Set("Upgrade", "WebSocket")
			return c.Do(req)
		},
		EarlyCheckResponse: func(proto string, res *Response) {
			if res.Proto != "HTTP/1.1" {
				t.Errorf("%s: expected HTTP/1.1, got %q", proto, res.Proto)
			}
			res.Proto = "HTTP/IGNORE" // skip later checks that Proto must be 1.1 vs 2.0
		},
	}.run(t)
}

func TestIdentityTransferEncoding(t *testing.T) { run(t, testIdentityTransferEncoding) }
func testIdentityTransferEncoding(t *testing.T, mode testMode) {
	const body = "body"
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		gotBody, _ := io.ReadAll(r.Body)
		if got, want := string(gotBody), body; got != want {
			t.Errorf("got request body = %q; want %q", got, want)
		}
		w.Header().Set("Transfer-Encoding", "identity")
		w.WriteHeader(StatusOK)
		w.(Flusher).Flush()
		io.WriteString(w, body)
	}))
	req, _ := NewRequest("GET", cst.ts.URL, strings.NewReader(body))
	res, err := cst.c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	gotBody, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(gotBody), body; got != want {
		t.Errorf("got response body = %q; want %q", got, want)
	}
}

func TestEarlyHintsRequest(t *testing.T) { run(t, testEarlyHintsRequest) }
func testEarlyHintsRequest(t *testing.T, mode testMode) {
	var wg sync.WaitGroup
	wg.Add(1)
	cst := newClientServerTest(t, mode, HandlerFunc(func(w ResponseWriter, r *Request) {
		h := w.Header()

		h.Add("Content-Length", "123") // must be ignored
		h.Add("Link", "</style.css>; rel=preload; as=style")
		h.Add("Link", "</script.js>; rel=preload; as=script")
		w.WriteHeader(StatusEarlyHints)

		wg.Wait()

		h.Add("Link", "</foo.js>; rel=preload; as=script")
		w.WriteHeader(StatusEarlyHints)

		w.Write([]byte("Hello"))
	}))

	checkLinkHeaders := func(t *testing.T, expected, got []string) {
		t.Helper()

		if len(expected) != len(got) {
			t.Errorf("got %d expected %d", len(got), len(expected))
		}

		for i := range expected {
			if expected[i] != got[i] {
				t.Errorf("got %q expected %q", got[i], expected[i])
			}
		}
	}

	checkExcludedHeaders := func(t *testing.T, header textproto.MIMEHeader) {
		t.Helper()

		for _, h := range []string{"Content-Length", "Transfer-Encoding"} {
			if v, ok := header[h]; ok {
				t.Errorf("%s is %q; must not be sent", h, v)
			}
		}
	}

	var respCounter uint8
	trace := &httptrace.ClientTrace{
		Got1xxResponse: func(code int, header textproto.MIMEHeader) error {
			switch respCounter {
			case 0:
				checkLinkHeaders(t, []string{"</style.css>; rel=preload; as=style", "</script.js>; rel=preload; as=script"}, header["Link"])
				checkExcludedHeaders(t, header)

				wg.Done()
			case 1:
				checkLinkHeaders(t, []string{"</style.css>; rel=preload; as=style", "</script.js>; rel=preload; as=script", "</foo.js>; rel=preload; as=script"}, header["Link"])
				checkExcludedHeaders(t, header)

			default:
				t.Error("Unexpected 1xx response")
			}

			respCounter++

			return nil
		},
	}
	req, _ := NewRequestWithContext(httptrace.WithClientTrace(context.Background(), trace), "GET", cst.ts.URL, nil)

	res, err := cst.c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()

	checkLinkHeaders(t, []string{"</style.css>; rel=preload; as=style", "</script.js>; rel=preload; as=script", "</foo.js>; rel=preload; as=script"}, res.Header["Link"])
	if cl := res.Header.Get("Content-Length"); cl != "123" {
		t.Errorf("Content-Length is %q; want 123", cl)
	}

	body, _ := io.ReadAll(res.Body)
	if string(body) != "Hello" {
		t.Errorf("Read body %q; want Hello", body)
	}
}
