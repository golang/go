// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests that use both the client & server, in both HTTP/1 and HTTP/2 mode.

package http_test

import (
	"bytes"
	"crypto/tls"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	. "net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
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

func newClientServerTest(t *testing.T, h2 bool, h Handler) *clientServerTest {
	cst := &clientServerTest{
		t:  t,
		h2: h2,
		h:  h,
		tr: &Transport{},
	}
	cst.c = &Client{Transport: cst.tr}
	if !h2 {
		cst.ts = httptest.NewServer(h)
		return cst
	}
	cst.ts = httptest.NewUnstartedServer(h)
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

func TestChunkedResponseHeaders_h1(t *testing.T) { testChunkedResponseHeaders(t, false) }
func TestChunkedResponseHeaders_h2(t *testing.T) { testChunkedResponseHeaders(t, true) }

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

// h12Compare is a test that compares HTTP/1 and HTTP/2 behavior
// against each other.
type h12Compare struct {
	Handler       func(ResponseWriter, *Request)                 // required
	ReqFunc       func(c *Client, url string) (*Response, error) // optional
	CheckResponse func(proto string, res *Response)              // optional
}

func (tt h12Compare) reqFunc() func(c *Client, url string) (*Response, error) {
	if tt.ReqFunc == nil {
		return (*Client).Get
	}
	return tt.ReqFunc
}

func (tt h12Compare) run(t *testing.T) {
	cst1 := newClientServerTest(t, false, HandlerFunc(tt.Handler))
	defer cst1.close()
	cst2 := newClientServerTest(t, true, HandlerFunc(tt.Handler))
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
	tt.normalizeRes(t, res1, "HTTP/1.1")
	tt.normalizeRes(t, res2, "HTTP/2.0")
	res1body, res2body := res1.Body, res2.Body
	res1.Body, res2.Body = nil, nil
	if !reflect.DeepEqual(res1, res2) {
		t.Errorf("Response headers to handler differed:\nhttp/1 (%v):\n\t%#v\nhttp/2 (%v):\n\t%#v",
			cst1.ts.URL, res1, cst2.ts.URL, res2)
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
	res.Request = nil
	if (res.TLS != nil) != (wantProto == "HTTP/2.0") {
		t.Errorf("%d. TLS set = %v; want %v", res.TLS != nil, res.TLS == nil)
	}
	res.TLS = nil
	// For now the HTTP/2 code isn't lying and saying
	// things are "chunked", since that's an HTTP/1.1
	// thing. I'd prefer not to lie and it shouldn't break
	// people.  I hope nobody's relying on that as a
	// heuristic for anything.
	if wantProto == "HTTP/2.0" && res.ContentLength == -1 && res.TransferEncoding == nil {
		res.TransferEncoding = []string{"chunked"}
	}
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
// writing more than they declared.  This test does not test whether
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

// TODO: TestH12_Trailers
// TODO: TestH12_AutoGzip (golang.org/issue/13298)

// Test304Responses verifies that 304s don't declare that they're
// chunking in their response headers and aren't allowed to produce
// output.
func Test304Responses_h1(t *testing.T) { test304Responses(t, false) }
func Test304Responses_h2(t *testing.T) { test304Responses(t, true) }

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
