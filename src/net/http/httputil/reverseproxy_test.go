// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reverse proxy tests.

package httputil

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

const fakeHopHeader = "X-Fake-Hop-Header-For-Test"

func init() {
	inOurTests = true
	hopHeaders = append(hopHeaders, fakeHopHeader)
}

func TestReverseProxy(t *testing.T) {
	const backendResponse = "I am the backend"
	const backendStatus = 404
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "GET" && r.FormValue("mode") == "hangup" {
			c, _, _ := w.(http.Hijacker).Hijack()
			c.Close()
			return
		}
		if len(r.TransferEncoding) > 0 {
			t.Errorf("backend got unexpected TransferEncoding: %v", r.TransferEncoding)
		}
		if r.Header.Get("X-Forwarded-For") == "" {
			t.Errorf("didn't get X-Forwarded-For header")
		}
		if c := r.Header.Get("Connection"); c != "" {
			t.Errorf("handler got Connection header value %q", c)
		}
		if c := r.Header.Get("Te"); c != "trailers" {
			t.Errorf("handler got Te header value %q; want 'trailers'", c)
		}
		if c := r.Header.Get("Upgrade"); c != "" {
			t.Errorf("handler got Upgrade header value %q", c)
		}
		if c := r.Header.Get("Proxy-Connection"); c != "" {
			t.Errorf("handler got Proxy-Connection header value %q", c)
		}
		if g, e := r.Host, "some-name"; g != e {
			t.Errorf("backend got Host header %q, want %q", g, e)
		}
		w.Header().Set("Trailers", "not a special header field name")
		w.Header().Set("Trailer", "X-Trailer")
		w.Header().Set("X-Foo", "bar")
		w.Header().Set("Upgrade", "foo")
		w.Header().Set(fakeHopHeader, "foo")
		w.Header().Add("X-Multi-Value", "foo")
		w.Header().Add("X-Multi-Value", "bar")
		http.SetCookie(w, &http.Cookie{Name: "flavor", Value: "chocolateChip"})
		w.WriteHeader(backendStatus)
		w.Write([]byte(backendResponse))
		w.Header().Set("X-Trailer", "trailer_value")
		w.Header().Set(http.TrailerPrefix+"X-Unannounced-Trailer", "unannounced_trailer_value")
	}))
	defer backend.Close()
	backendURL, err := url.Parse(backend.URL)
	if err != nil {
		t.Fatal(err)
	}
	proxyHandler := NewSingleHostReverseProxy(backendURL)
	proxyHandler.ErrorLog = log.New(io.Discard, "", 0) // quiet for tests
	frontend := httptest.NewServer(proxyHandler)
	defer frontend.Close()
	frontendClient := frontend.Client()

	getReq, _ := http.NewRequest("GET", frontend.URL, nil)
	getReq.Host = "some-name"
	getReq.Header.Set("Connection", "close")
	getReq.Header.Set("Te", "trailers")
	getReq.Header.Set("Proxy-Connection", "should be deleted")
	getReq.Header.Set("Upgrade", "foo")
	getReq.Close = true
	res, err := frontendClient.Do(getReq)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if g, e := res.StatusCode, backendStatus; g != e {
		t.Errorf("got res.StatusCode %d; expected %d", g, e)
	}
	if g, e := res.Header.Get("X-Foo"), "bar"; g != e {
		t.Errorf("got X-Foo %q; expected %q", g, e)
	}
	if c := res.Header.Get(fakeHopHeader); c != "" {
		t.Errorf("got %s header value %q", fakeHopHeader, c)
	}
	if g, e := res.Header.Get("Trailers"), "not a special header field name"; g != e {
		t.Errorf("header Trailers = %q; want %q", g, e)
	}
	if g, e := len(res.Header["X-Multi-Value"]), 2; g != e {
		t.Errorf("got %d X-Multi-Value header values; expected %d", g, e)
	}
	if g, e := len(res.Header["Set-Cookie"]), 1; g != e {
		t.Fatalf("got %d SetCookies, want %d", g, e)
	}
	if g, e := res.Trailer, (http.Header{"X-Trailer": nil}); !reflect.DeepEqual(g, e) {
		t.Errorf("before reading body, Trailer = %#v; want %#v", g, e)
	}
	if cookie := res.Cookies()[0]; cookie.Name != "flavor" {
		t.Errorf("unexpected cookie %q", cookie.Name)
	}
	bodyBytes, _ := io.ReadAll(res.Body)
	if g, e := string(bodyBytes), backendResponse; g != e {
		t.Errorf("got body %q; expected %q", g, e)
	}
	if g, e := res.Trailer.Get("X-Trailer"), "trailer_value"; g != e {
		t.Errorf("Trailer(X-Trailer) = %q ; want %q", g, e)
	}
	if g, e := res.Trailer.Get("X-Unannounced-Trailer"), "unannounced_trailer_value"; g != e {
		t.Errorf("Trailer(X-Unannounced-Trailer) = %q ; want %q", g, e)
	}

	// Test that a backend failing to be reached or one which doesn't return
	// a response results in a StatusBadGateway.
	getReq, _ = http.NewRequest("GET", frontend.URL+"/?mode=hangup", nil)
	getReq.Close = true
	res, err = frontendClient.Do(getReq)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.StatusCode != http.StatusBadGateway {
		t.Errorf("request to bad proxy = %v; want 502 StatusBadGateway", res.Status)
	}

}

// Issue 16875: remove any proxied headers mentioned in the "Connection"
// header value.
func TestReverseProxyStripHeadersPresentInConnection(t *testing.T) {
	const fakeConnectionToken = "X-Fake-Connection-Token"
	const backendResponse = "I am the backend"

	// someConnHeader is some arbitrary header to be declared as a hop-by-hop header
	// in the Request's Connection header.
	const someConnHeader = "X-Some-Conn-Header"

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if c := r.Header.Get("Connection"); c != "" {
			t.Errorf("handler got header %q = %q; want empty", "Connection", c)
		}
		if c := r.Header.Get(fakeConnectionToken); c != "" {
			t.Errorf("handler got header %q = %q; want empty", fakeConnectionToken, c)
		}
		if c := r.Header.Get(someConnHeader); c != "" {
			t.Errorf("handler got header %q = %q; want empty", someConnHeader, c)
		}
		w.Header().Add("Connection", "Upgrade, "+fakeConnectionToken)
		w.Header().Add("Connection", someConnHeader)
		w.Header().Set(someConnHeader, "should be deleted")
		w.Header().Set(fakeConnectionToken, "should be deleted")
		io.WriteString(w, backendResponse)
	}))
	defer backend.Close()
	backendURL, err := url.Parse(backend.URL)
	if err != nil {
		t.Fatal(err)
	}
	proxyHandler := NewSingleHostReverseProxy(backendURL)
	frontend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxyHandler.ServeHTTP(w, r)
		if c := r.Header.Get(someConnHeader); c != "should be deleted" {
			t.Errorf("handler modified header %q = %q; want %q", someConnHeader, c, "should be deleted")
		}
		if c := r.Header.Get(fakeConnectionToken); c != "should be deleted" {
			t.Errorf("handler modified header %q = %q; want %q", fakeConnectionToken, c, "should be deleted")
		}
		c := r.Header["Connection"]
		var cf []string
		for _, f := range c {
			for _, sf := range strings.Split(f, ",") {
				if sf = strings.TrimSpace(sf); sf != "" {
					cf = append(cf, sf)
				}
			}
		}
		sort.Strings(cf)
		expectedValues := []string{"Upgrade", someConnHeader, fakeConnectionToken}
		sort.Strings(expectedValues)
		if !reflect.DeepEqual(cf, expectedValues) {
			t.Errorf("handler modified header %q = %q; want %q", "Connection", cf, expectedValues)
		}
	}))
	defer frontend.Close()

	getReq, _ := http.NewRequest("GET", frontend.URL, nil)
	getReq.Header.Add("Connection", "Upgrade, "+fakeConnectionToken)
	getReq.Header.Add("Connection", someConnHeader)
	getReq.Header.Set(someConnHeader, "should be deleted")
	getReq.Header.Set(fakeConnectionToken, "should be deleted")
	res, err := frontend.Client().Do(getReq)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	defer res.Body.Close()
	bodyBytes, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatalf("reading body: %v", err)
	}
	if got, want := string(bodyBytes), backendResponse; got != want {
		t.Errorf("got body %q; want %q", got, want)
	}
	if c := res.Header.Get("Connection"); c != "" {
		t.Errorf("handler got header %q = %q; want empty", "Connection", c)
	}
	if c := res.Header.Get(someConnHeader); c != "" {
		t.Errorf("handler got header %q = %q; want empty", someConnHeader, c)
	}
	if c := res.Header.Get(fakeConnectionToken); c != "" {
		t.Errorf("handler got header %q = %q; want empty", fakeConnectionToken, c)
	}
}

func TestXForwardedFor(t *testing.T) {
	const prevForwardedFor = "client ip"
	const backendResponse = "I am the backend"
	const backendStatus = 404
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Forwarded-For") == "" {
			t.Errorf("didn't get X-Forwarded-For header")
		}
		if !strings.Contains(r.Header.Get("X-Forwarded-For"), prevForwardedFor) {
			t.Errorf("X-Forwarded-For didn't contain prior data")
		}
		w.WriteHeader(backendStatus)
		w.Write([]byte(backendResponse))
	}))
	defer backend.Close()
	backendURL, err := url.Parse(backend.URL)
	if err != nil {
		t.Fatal(err)
	}
	proxyHandler := NewSingleHostReverseProxy(backendURL)
	frontend := httptest.NewServer(proxyHandler)
	defer frontend.Close()

	getReq, _ := http.NewRequest("GET", frontend.URL, nil)
	getReq.Host = "some-name"
	getReq.Header.Set("Connection", "close")
	getReq.Header.Set("X-Forwarded-For", prevForwardedFor)
	getReq.Close = true
	res, err := frontend.Client().Do(getReq)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if g, e := res.StatusCode, backendStatus; g != e {
		t.Errorf("got res.StatusCode %d; expected %d", g, e)
	}
	bodyBytes, _ := io.ReadAll(res.Body)
	if g, e := string(bodyBytes), backendResponse; g != e {
		t.Errorf("got body %q; expected %q", g, e)
	}
}

// Issue 38079: don't append to X-Forwarded-For if it's present but nil
func TestXForwardedFor_Omit(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if v := r.Header.Get("X-Forwarded-For"); v != "" {
			t.Errorf("got X-Forwarded-For header: %q", v)
		}
		w.Write([]byte("hi"))
	}))
	defer backend.Close()
	backendURL, err := url.Parse(backend.URL)
	if err != nil {
		t.Fatal(err)
	}
	proxyHandler := NewSingleHostReverseProxy(backendURL)
	frontend := httptest.NewServer(proxyHandler)
	defer frontend.Close()

	oldDirector := proxyHandler.Director
	proxyHandler.Director = func(r *http.Request) {
		r.Header["X-Forwarded-For"] = nil
		oldDirector(r)
	}

	getReq, _ := http.NewRequest("GET", frontend.URL, nil)
	getReq.Host = "some-name"
	getReq.Close = true
	res, err := frontend.Client().Do(getReq)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	res.Body.Close()
}

var proxyQueryTests = []struct {
	baseSuffix string // suffix to add to backend URL
	reqSuffix  string // suffix to add to frontend's request URL
	want       string // what backend should see for final request URL (without ?)
}{
	{"", "", ""},
	{"?sta=tic", "?us=er", "sta=tic&us=er"},
	{"", "?us=er", "us=er"},
	{"?sta=tic", "", "sta=tic"},
}

func TestReverseProxyQuery(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Got-Query", r.URL.RawQuery)
		w.Write([]byte("hi"))
	}))
	defer backend.Close()

	for i, tt := range proxyQueryTests {
		backendURL, err := url.Parse(backend.URL + tt.baseSuffix)
		if err != nil {
			t.Fatal(err)
		}
		frontend := httptest.NewServer(NewSingleHostReverseProxy(backendURL))
		req, _ := http.NewRequest("GET", frontend.URL+tt.reqSuffix, nil)
		req.Close = true
		res, err := frontend.Client().Do(req)
		if err != nil {
			t.Fatalf("%d. Get: %v", i, err)
		}
		if g, e := res.Header.Get("X-Got-Query"), tt.want; g != e {
			t.Errorf("%d. got query %q; expected %q", i, g, e)
		}
		res.Body.Close()
		frontend.Close()
	}
}

func TestReverseProxyFlushInterval(t *testing.T) {
	const expected = "hi"
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(expected))
	}))
	defer backend.Close()

	backendURL, err := url.Parse(backend.URL)
	if err != nil {
		t.Fatal(err)
	}

	proxyHandler := NewSingleHostReverseProxy(backendURL)
	proxyHandler.FlushInterval = time.Microsecond

	frontend := httptest.NewServer(proxyHandler)
	defer frontend.Close()

	req, _ := http.NewRequest("GET", frontend.URL, nil)
	req.Close = true
	res, err := frontend.Client().Do(req)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	defer res.Body.Close()
	if bodyBytes, _ := io.ReadAll(res.Body); string(bodyBytes) != expected {
		t.Errorf("got body %q; expected %q", bodyBytes, expected)
	}
}

func TestReverseProxyFlushIntervalHeaders(t *testing.T) {
	const expected = "hi"
	stopCh := make(chan struct{})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Add("MyHeader", expected)
		w.WriteHeader(200)
		w.(http.Flusher).Flush()
		<-stopCh
	}))
	defer backend.Close()
	defer close(stopCh)

	backendURL, err := url.Parse(backend.URL)
	if err != nil {
		t.Fatal(err)
	}

	proxyHandler := NewSingleHostReverseProxy(backendURL)
	proxyHandler.FlushInterval = time.Microsecond

	frontend := httptest.NewServer(proxyHandler)
	defer frontend.Close()

	req, _ := http.NewRequest("GET", frontend.URL, nil)
	req.Close = true

	ctx, cancel := context.WithTimeout(req.Context(), 10*time.Second)
	defer cancel()
	req = req.WithContext(ctx)

	res, err := frontend.Client().Do(req)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	defer res.Body.Close()

	if res.Header.Get("MyHeader") != expected {
		t.Errorf("got header %q; expected %q", res.Header.Get("MyHeader"), expected)
	}
}

func TestReverseProxyCancellation(t *testing.T) {
	const backendResponse = "I am the backend"

	reqInFlight := make(chan struct{})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(reqInFlight) // cause the client to cancel its request

		select {
		case <-time.After(10 * time.Second):
			// Note: this should only happen in broken implementations, and the
			// closenotify case should be instantaneous.
			t.Error("Handler never saw CloseNotify")
			return
		case <-w.(http.CloseNotifier).CloseNotify():
		}

		w.WriteHeader(http.StatusOK)
		w.Write([]byte(backendResponse))
	}))

	defer backend.Close()

	backend.Config.ErrorLog = log.New(io.Discard, "", 0)

	backendURL, err := url.Parse(backend.URL)
	if err != nil {
		t.Fatal(err)
	}

	proxyHandler := NewSingleHostReverseProxy(backendURL)

	// Discards errors of the form:
	// http: proxy error: read tcp 127.0.0.1:44643: use of closed network connection
	proxyHandler.ErrorLog = log.New(io.Discard, "", 0)

	frontend := httptest.NewServer(proxyHandler)
	defer frontend.Close()
	frontendClient := frontend.Client()

	getReq, _ := http.NewRequest("GET", frontend.URL, nil)
	go func() {
		<-reqInFlight
		frontendClient.Transport.(*http.Transport).CancelRequest(getReq)
	}()
	res, err := frontendClient.Do(getReq)
	if res != nil {
		t.Errorf("got response %v; want nil", res.Status)
	}
	if err == nil {
		// This should be an error like:
		// Get "http://127.0.0.1:58079": read tcp 127.0.0.1:58079:
		//    use of closed network connection
		t.Error("Server.Client().Do() returned nil error; want non-nil error")
	}
}

func req(t *testing.T, v string) *http.Request {
	req, err := http.ReadRequest(bufio.NewReader(strings.NewReader(v)))
	if err != nil {
		t.Fatal(err)
	}
	return req
}

// Issue 12344
func TestNilBody(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("hi"))
	}))
	defer backend.Close()

	frontend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		backURL, _ := url.Parse(backend.URL)
		rp := NewSingleHostReverseProxy(backURL)
		r := req(t, "GET / HTTP/1.0\r\n\r\n")
		r.Body = nil // this accidentally worked in Go 1.4 and below, so keep it working
		rp.ServeHTTP(w, r)
	}))
	defer frontend.Close()

	res, err := http.Get(frontend.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	slurp, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(slurp) != "hi" {
		t.Errorf("Got %q; want %q", slurp, "hi")
	}
}

// Issue 15524
func TestUserAgentHeader(t *testing.T) {
	const explicitUA = "explicit UA"
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/noua" {
			if c := r.Header.Get("User-Agent"); c != "" {
				t.Errorf("handler got non-empty User-Agent header %q", c)
			}
			return
		}
		if c := r.Header.Get("User-Agent"); c != explicitUA {
			t.Errorf("handler got unexpected User-Agent header %q", c)
		}
	}))
	defer backend.Close()
	backendURL, err := url.Parse(backend.URL)
	if err != nil {
		t.Fatal(err)
	}
	proxyHandler := NewSingleHostReverseProxy(backendURL)
	proxyHandler.ErrorLog = log.New(io.Discard, "", 0) // quiet for tests
	frontend := httptest.NewServer(proxyHandler)
	defer frontend.Close()
	frontendClient := frontend.Client()

	getReq, _ := http.NewRequest("GET", frontend.URL, nil)
	getReq.Header.Set("User-Agent", explicitUA)
	getReq.Close = true
	res, err := frontendClient.Do(getReq)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	res.Body.Close()

	getReq, _ = http.NewRequest("GET", frontend.URL+"/noua", nil)
	getReq.Header.Set("User-Agent", "")
	getReq.Close = true
	res, err = frontendClient.Do(getReq)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	res.Body.Close()
}

type bufferPool struct {
	get func() []byte
	put func([]byte)
}

func (bp bufferPool) Get() []byte  { return bp.get() }
func (bp bufferPool) Put(v []byte) { bp.put(v) }

func TestReverseProxyGetPutBuffer(t *testing.T) {
	const msg = "hi"
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, msg)
	}))
	defer backend.Close()

	backendURL, err := url.Parse(backend.URL)
	if err != nil {
		t.Fatal(err)
	}

	var (
		mu  sync.Mutex
		log []string
	)
	addLog := func(event string) {
		mu.Lock()
		defer mu.Unlock()
		log = append(log, event)
	}
	rp := NewSingleHostReverseProxy(backendURL)
	const size = 1234
	rp.BufferPool = bufferPool{
		get: func() []byte {
			addLog("getBuf")
			return make([]byte, size)
		},
		put: func(p []byte) {
			addLog("putBuf-" + strconv.Itoa(len(p)))
		},
	}
	frontend := httptest.NewServer(rp)
	defer frontend.Close()

	req, _ := http.NewRequest("GET", frontend.URL, nil)
	req.Close = true
	res, err := frontend.Client().Do(req)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	slurp, err := io.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		t.Fatalf("reading body: %v", err)
	}
	if string(slurp) != msg {
		t.Errorf("msg = %q; want %q", slurp, msg)
	}
	wantLog := []string{"getBuf", "putBuf-" + strconv.Itoa(size)}
	mu.Lock()
	defer mu.Unlock()
	if !reflect.DeepEqual(log, wantLog) {
		t.Errorf("Log events = %q; want %q", log, wantLog)
	}
}

func TestReverseProxy_Post(t *testing.T) {
	const backendResponse = "I am the backend"
	const backendStatus = 200
	var requestBody = bytes.Repeat([]byte("a"), 1<<20)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		slurp, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("Backend body read = %v", err)
		}
		if len(slurp) != len(requestBody) {
			t.Errorf("Backend read %d request body bytes; want %d", len(slurp), len(requestBody))
		}
		if !bytes.Equal(slurp, requestBody) {
			t.Error("Backend read wrong request body.") // 1MB; omitting details
		}
		w.Write([]byte(backendResponse))
	}))
	defer backend.Close()
	backendURL, err := url.Parse(backend.URL)
	if err != nil {
		t.Fatal(err)
	}
	proxyHandler := NewSingleHostReverseProxy(backendURL)
	frontend := httptest.NewServer(proxyHandler)
	defer frontend.Close()

	postReq, _ := http.NewRequest("POST", frontend.URL, bytes.NewReader(requestBody))
	res, err := frontend.Client().Do(postReq)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	if g, e := res.StatusCode, backendStatus; g != e {
		t.Errorf("got res.StatusCode %d; expected %d", g, e)
	}
	bodyBytes, _ := io.ReadAll(res.Body)
	if g, e := string(bodyBytes), backendResponse; g != e {
		t.Errorf("got body %q; expected %q", g, e)
	}
}

type RoundTripperFunc func(*http.Request) (*http.Response, error)

func (fn RoundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}

// Issue 16036: send a Request with a nil Body when possible
func TestReverseProxy_NilBody(t *testing.T) {
	backendURL, _ := url.Parse("http://fake.tld/")
	proxyHandler := NewSingleHostReverseProxy(backendURL)
	proxyHandler.ErrorLog = log.New(io.Discard, "", 0) // quiet for tests
	proxyHandler.Transport = RoundTripperFunc(func(req *http.Request) (*http.Response, error) {
		if req.Body != nil {
			t.Error("Body != nil; want a nil Body")
		}
		return nil, errors.New("done testing the interesting part; so force a 502 Gateway error")
	})
	frontend := httptest.NewServer(proxyHandler)
	defer frontend.Close()

	res, err := frontend.Client().Get(frontend.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != 502 {
		t.Errorf("status code = %v; want 502 (Gateway Error)", res.Status)
	}
}

// Issue 33142: always allocate the request headers
func TestReverseProxy_AllocatedHeader(t *testing.T) {
	proxyHandler := new(ReverseProxy)
	proxyHandler.ErrorLog = log.New(io.Discard, "", 0) // quiet for tests
	proxyHandler.Director = func(*http.Request) {}     // noop
	proxyHandler.Transport = RoundTripperFunc(func(req *http.Request) (*http.Response, error) {
		if req.Header == nil {
			t.Error("Header == nil; want a non-nil Header")
		}
		return nil, errors.New("done testing the interesting part; so force a 502 Gateway error")
	})

	proxyHandler.ServeHTTP(httptest.NewRecorder(), &http.Request{
		Method:     "GET",
		URL:        &url.URL{Scheme: "http", Host: "fake.tld", Path: "/"},
		Proto:      "HTTP/1.0",
		ProtoMajor: 1,
	})
}

// Issue 14237. Test ModifyResponse and that an error from it
// causes the proxy to return StatusBadGateway, or StatusOK otherwise.
func TestReverseProxyModifyResponse(t *testing.T) {
	backendServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Add("X-Hit-Mod", fmt.Sprintf("%v", r.URL.Path == "/mod"))
	}))
	defer backendServer.Close()

	rpURL, _ := url.Parse(backendServer.URL)
	rproxy := NewSingleHostReverseProxy(rpURL)
	rproxy.ErrorLog = log.New(io.Discard, "", 0) // quiet for tests
	rproxy.ModifyResponse = func(resp *http.Response) error {
		if resp.Header.Get("X-Hit-Mod") != "true" {
			return fmt.Errorf("tried to by-pass proxy")
		}
		return nil
	}

	frontendProxy := httptest.NewServer(rproxy)
	defer frontendProxy.Close()

	tests := []struct {
		url      string
		wantCode int
	}{
		{frontendProxy.URL + "/mod", http.StatusOK},
		{frontendProxy.URL + "/schedule", http.StatusBadGateway},
	}

	for i, tt := range tests {
		resp, err := http.Get(tt.url)
		if err != nil {
			t.Fatalf("failed to reach proxy: %v", err)
		}
		if g, e := resp.StatusCode, tt.wantCode; g != e {
			t.Errorf("#%d: got res.StatusCode %d; expected %d", i, g, e)
		}
		resp.Body.Close()
	}
}

type failingRoundTripper struct{}

func (failingRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, errors.New("some error")
}

type staticResponseRoundTripper struct{ res *http.Response }

func (rt staticResponseRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	return rt.res, nil
}

func TestReverseProxyErrorHandler(t *testing.T) {
	tests := []struct {
		name           string
		wantCode       int
		errorHandler   func(http.ResponseWriter, *http.Request, error)
		transport      http.RoundTripper // defaults to failingRoundTripper
		modifyResponse func(*http.Response) error
	}{
		{
			name:     "default",
			wantCode: http.StatusBadGateway,
		},
		{
			name:         "errorhandler",
			wantCode:     http.StatusTeapot,
			errorHandler: func(rw http.ResponseWriter, req *http.Request, err error) { rw.WriteHeader(http.StatusTeapot) },
		},
		{
			name: "modifyresponse_noerr",
			transport: staticResponseRoundTripper{
				&http.Response{StatusCode: 345, Body: http.NoBody},
			},
			modifyResponse: func(res *http.Response) error {
				res.StatusCode++
				return nil
			},
			errorHandler: func(rw http.ResponseWriter, req *http.Request, err error) { rw.WriteHeader(http.StatusTeapot) },
			wantCode:     346,
		},
		{
			name: "modifyresponse_err",
			transport: staticResponseRoundTripper{
				&http.Response{StatusCode: 345, Body: http.NoBody},
			},
			modifyResponse: func(res *http.Response) error {
				res.StatusCode++
				return errors.New("some error to trigger errorHandler")
			},
			errorHandler: func(rw http.ResponseWriter, req *http.Request, err error) { rw.WriteHeader(http.StatusTeapot) },
			wantCode:     http.StatusTeapot,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			target := &url.URL{
				Scheme: "http",
				Host:   "dummy.tld",
				Path:   "/",
			}
			rproxy := NewSingleHostReverseProxy(target)
			rproxy.Transport = tt.transport
			rproxy.ModifyResponse = tt.modifyResponse
			if rproxy.Transport == nil {
				rproxy.Transport = failingRoundTripper{}
			}
			rproxy.ErrorLog = log.New(io.Discard, "", 0) // quiet for tests
			if tt.errorHandler != nil {
				rproxy.ErrorHandler = tt.errorHandler
			}
			frontendProxy := httptest.NewServer(rproxy)
			defer frontendProxy.Close()

			resp, err := http.Get(frontendProxy.URL + "/test")
			if err != nil {
				t.Fatalf("failed to reach proxy: %v", err)
			}
			if g, e := resp.StatusCode, tt.wantCode; g != e {
				t.Errorf("got res.StatusCode %d; expected %d", g, e)
			}
			resp.Body.Close()
		})
	}
}

// Issue 16659: log errors from short read
func TestReverseProxy_CopyBuffer(t *testing.T) {
	backendServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		out := "this call was relayed by the reverse proxy"
		// Coerce a wrong content length to induce io.UnexpectedEOF
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(out)*2))
		fmt.Fprintln(w, out)
	}))
	defer backendServer.Close()

	rpURL, err := url.Parse(backendServer.URL)
	if err != nil {
		t.Fatal(err)
	}

	var proxyLog bytes.Buffer
	rproxy := NewSingleHostReverseProxy(rpURL)
	rproxy.ErrorLog = log.New(&proxyLog, "", log.Lshortfile)
	donec := make(chan bool, 1)
	frontendProxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() { donec <- true }()
		rproxy.ServeHTTP(w, r)
	}))
	defer frontendProxy.Close()

	if _, err = frontendProxy.Client().Get(frontendProxy.URL); err == nil {
		t.Fatalf("want non-nil error")
	}
	// The race detector complains about the proxyLog usage in logf in copyBuffer
	// and our usage below with proxyLog.Bytes() so we're explicitly using a
	// channel to ensure that the ReverseProxy's ServeHTTP is done before we
	// continue after Get.
	<-donec

	expected := []string{
		"EOF",
		"read",
	}
	for _, phrase := range expected {
		if !bytes.Contains(proxyLog.Bytes(), []byte(phrase)) {
			t.Errorf("expected log to contain phrase %q", phrase)
		}
	}
}

type staticTransport struct {
	res *http.Response
}

func (t *staticTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return t.res, nil
}

func BenchmarkServeHTTP(b *testing.B) {
	res := &http.Response{
		StatusCode: 200,
		Body:       io.NopCloser(strings.NewReader("")),
	}
	proxy := &ReverseProxy{
		Director:  func(*http.Request) {},
		Transport: &staticTransport{res},
	}

	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/", nil)

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		proxy.ServeHTTP(w, r)
	}
}

func TestServeHTTPDeepCopy(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("Hello Gopher!"))
	}))
	defer backend.Close()
	backendURL, err := url.Parse(backend.URL)
	if err != nil {
		t.Fatal(err)
	}

	type result struct {
		before, after string
	}

	resultChan := make(chan result, 1)
	proxyHandler := NewSingleHostReverseProxy(backendURL)
	frontend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		before := r.URL.String()
		proxyHandler.ServeHTTP(w, r)
		after := r.URL.String()
		resultChan <- result{before: before, after: after}
	}))
	defer frontend.Close()

	want := result{before: "/", after: "/"}

	res, err := frontend.Client().Get(frontend.URL)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	res.Body.Close()

	got := <-resultChan
	if got != want {
		t.Errorf("got = %+v; want = %+v", got, want)
	}
}

// Issue 18327: verify we always do a deep copy of the Request.Header map
// before any mutations.
func TestClonesRequestHeaders(t *testing.T) {
	log.SetOutput(io.Discard)
	defer log.SetOutput(os.Stderr)
	req, _ := http.NewRequest("GET", "http://foo.tld/", nil)
	req.RemoteAddr = "1.2.3.4:56789"
	rp := &ReverseProxy{
		Director: func(req *http.Request) {
			req.Header.Set("From-Director", "1")
		},
		Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			if v := req.Header.Get("From-Director"); v != "1" {
				t.Errorf("From-Directory value = %q; want 1", v)
			}
			return nil, io.EOF
		}),
	}
	rp.ServeHTTP(httptest.NewRecorder(), req)

	if req.Header.Get("From-Director") == "1" {
		t.Error("Director header mutation modified caller's request")
	}
	if req.Header.Get("X-Forwarded-For") != "" {
		t.Error("X-Forward-For header mutation modified caller's request")
	}

}

type roundTripperFunc func(req *http.Request) (*http.Response, error)

func (fn roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}

func TestModifyResponseClosesBody(t *testing.T) {
	req, _ := http.NewRequest("GET", "http://foo.tld/", nil)
	req.RemoteAddr = "1.2.3.4:56789"
	closeCheck := new(checkCloser)
	logBuf := new(bytes.Buffer)
	outErr := errors.New("ModifyResponse error")
	rp := &ReverseProxy{
		Director: func(req *http.Request) {},
		Transport: &staticTransport{&http.Response{
			StatusCode: 200,
			Body:       closeCheck,
		}},
		ErrorLog: log.New(logBuf, "", 0),
		ModifyResponse: func(*http.Response) error {
			return outErr
		},
	}
	rec := httptest.NewRecorder()
	rp.ServeHTTP(rec, req)
	res := rec.Result()
	if g, e := res.StatusCode, http.StatusBadGateway; g != e {
		t.Errorf("got res.StatusCode %d; expected %d", g, e)
	}
	if !closeCheck.closed {
		t.Errorf("body should have been closed")
	}
	if g, e := logBuf.String(), outErr.Error(); !strings.Contains(g, e) {
		t.Errorf("ErrorLog %q does not contain %q", g, e)
	}
}

type checkCloser struct {
	closed bool
}

func (cc *checkCloser) Close() error {
	cc.closed = true
	return nil
}

func (cc *checkCloser) Read(b []byte) (int, error) {
	return len(b), nil
}

// Issue 23643: panic on body copy error
func TestReverseProxy_PanicBodyError(t *testing.T) {
	log.SetOutput(io.Discard)
	defer log.SetOutput(os.Stderr)
	backendServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		out := "this call was relayed by the reverse proxy"
		// Coerce a wrong content length to induce io.ErrUnexpectedEOF
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(out)*2))
		fmt.Fprintln(w, out)
	}))
	defer backendServer.Close()

	rpURL, err := url.Parse(backendServer.URL)
	if err != nil {
		t.Fatal(err)
	}

	rproxy := NewSingleHostReverseProxy(rpURL)

	// Ensure that the handler panics when the body read encounters an
	// io.ErrUnexpectedEOF
	defer func() {
		err := recover()
		if err == nil {
			t.Fatal("handler should have panicked")
		}
		if err != http.ErrAbortHandler {
			t.Fatal("expected ErrAbortHandler, got", err)
		}
	}()
	req, _ := http.NewRequest("GET", "http://foo.tld/", nil)
	rproxy.ServeHTTP(httptest.NewRecorder(), req)
}

func TestSelectFlushInterval(t *testing.T) {
	tests := []struct {
		name string
		p    *ReverseProxy
		res  *http.Response
		want time.Duration
	}{
		{
			name: "default",
			res:  &http.Response{},
			p:    &ReverseProxy{FlushInterval: 123},
			want: 123,
		},
		{
			name: "server-sent events overrides non-zero",
			res: &http.Response{
				Header: http.Header{
					"Content-Type": {"text/event-stream"},
				},
			},
			p:    &ReverseProxy{FlushInterval: 123},
			want: -1,
		},
		{
			name: "server-sent events overrides zero",
			res: &http.Response{
				Header: http.Header{
					"Content-Type": {"text/event-stream"},
				},
			},
			p:    &ReverseProxy{FlushInterval: 0},
			want: -1,
		},
		{
			name: "Content-Length: -1, overrides non-zero",
			res: &http.Response{
				ContentLength: -1,
			},
			p:    &ReverseProxy{FlushInterval: 123},
			want: -1,
		},
		{
			name: "Content-Length: -1, overrides zero",
			res: &http.Response{
				ContentLength: -1,
			},
			p:    &ReverseProxy{FlushInterval: 0},
			want: -1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.p.flushInterval(tt.res)
			if got != tt.want {
				t.Errorf("flushLatency = %v; want %v", got, tt.want)
			}
		})
	}
}

func TestReverseProxyWebSocket(t *testing.T) {
	backendServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if upgradeType(r.Header) != "websocket" {
			t.Error("unexpected backend request")
			http.Error(w, "unexpected request", 400)
			return
		}
		c, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		defer c.Close()
		io.WriteString(c, "HTTP/1.1 101 Switching Protocols\r\nConnection: upgrade\r\nUpgrade: WebSocket\r\n\r\n")
		bs := bufio.NewScanner(c)
		if !bs.Scan() {
			t.Errorf("backend failed to read line from client: %v", bs.Err())
			return
		}
		fmt.Fprintf(c, "backend got %q\n", bs.Text())
	}))
	defer backendServer.Close()

	backURL, _ := url.Parse(backendServer.URL)
	rproxy := NewSingleHostReverseProxy(backURL)
	rproxy.ErrorLog = log.New(io.Discard, "", 0) // quiet for tests
	rproxy.ModifyResponse = func(res *http.Response) error {
		res.Header.Add("X-Modified", "true")
		return nil
	}

	handler := http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		rw.Header().Set("X-Header", "X-Value")
		rproxy.ServeHTTP(rw, req)
		if got, want := rw.Header().Get("X-Modified"), "true"; got != want {
			t.Errorf("response writer X-Modified header = %q; want %q", got, want)
		}
	})

	frontendProxy := httptest.NewServer(handler)
	defer frontendProxy.Close()

	req, _ := http.NewRequest("GET", frontendProxy.URL, nil)
	req.Header.Set("Connection", "Upgrade")
	req.Header.Set("Upgrade", "websocket")

	c := frontendProxy.Client()
	res, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != 101 {
		t.Fatalf("status = %v; want 101", res.Status)
	}

	got := res.Header.Get("X-Header")
	want := "X-Value"
	if got != want {
		t.Errorf("Header(XHeader) = %q; want %q", got, want)
	}

	if upgradeType(res.Header) != "websocket" {
		t.Fatalf("not websocket upgrade; got %#v", res.Header)
	}
	rwc, ok := res.Body.(io.ReadWriteCloser)
	if !ok {
		t.Fatalf("response body is of type %T; does not implement ReadWriteCloser", res.Body)
	}
	defer rwc.Close()

	if got, want := res.Header.Get("X-Modified"), "true"; got != want {
		t.Errorf("response X-Modified header = %q; want %q", got, want)
	}

	io.WriteString(rwc, "Hello\n")
	bs := bufio.NewScanner(rwc)
	if !bs.Scan() {
		t.Fatalf("Scan: %v", bs.Err())
	}
	got = bs.Text()
	want = `backend got "Hello"`
	if got != want {
		t.Errorf("got %#q, want %#q", got, want)
	}
}

func TestReverseProxyWebSocketCancelation(t *testing.T) {
	n := 5
	triggerCancelCh := make(chan bool, n)
	nthResponse := func(i int) string {
		return fmt.Sprintf("backend response #%d\n", i)
	}
	terminalMsg := "final message"

	cst := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if g, ws := upgradeType(r.Header), "websocket"; g != ws {
			t.Errorf("Unexpected upgrade type %q, want %q", g, ws)
			http.Error(w, "Unexpected request", 400)
			return
		}
		conn, bufrw, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		defer conn.Close()

		upgradeMsg := "HTTP/1.1 101 Switching Protocols\r\nConnection: upgrade\r\nUpgrade: WebSocket\r\n\r\n"
		if _, err := io.WriteString(conn, upgradeMsg); err != nil {
			t.Error(err)
			return
		}
		if _, _, err := bufrw.ReadLine(); err != nil {
			t.Errorf("Failed to read line from client: %v", err)
			return
		}

		for i := 0; i < n; i++ {
			if _, err := bufrw.WriteString(nthResponse(i)); err != nil {
				select {
				case <-triggerCancelCh:
				default:
					t.Errorf("Writing response #%d failed: %v", i, err)
				}
				return
			}
			bufrw.Flush()
			time.Sleep(time.Second)
		}
		if _, err := bufrw.WriteString(terminalMsg); err != nil {
			select {
			case <-triggerCancelCh:
			default:
				t.Errorf("Failed to write terminal message: %v", err)
			}
		}
		bufrw.Flush()
	}))
	defer cst.Close()

	backendURL, _ := url.Parse(cst.URL)
	rproxy := NewSingleHostReverseProxy(backendURL)
	rproxy.ErrorLog = log.New(io.Discard, "", 0) // quiet for tests
	rproxy.ModifyResponse = func(res *http.Response) error {
		res.Header.Add("X-Modified", "true")
		return nil
	}

	handler := http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		rw.Header().Set("X-Header", "X-Value")
		ctx, cancel := context.WithCancel(req.Context())
		go func() {
			<-triggerCancelCh
			cancel()
		}()
		rproxy.ServeHTTP(rw, req.WithContext(ctx))
	})

	frontendProxy := httptest.NewServer(handler)
	defer frontendProxy.Close()

	req, _ := http.NewRequest("GET", frontendProxy.URL, nil)
	req.Header.Set("Connection", "Upgrade")
	req.Header.Set("Upgrade", "websocket")

	res, err := frontendProxy.Client().Do(req)
	if err != nil {
		t.Fatalf("Dialing to frontend proxy: %v", err)
	}
	defer res.Body.Close()
	if g, w := res.StatusCode, 101; g != w {
		t.Fatalf("Switching protocols failed, got: %d, want: %d", g, w)
	}

	if g, w := res.Header.Get("X-Header"), "X-Value"; g != w {
		t.Errorf("X-Header mismatch\n\tgot:  %q\n\twant: %q", g, w)
	}

	if g, w := upgradeType(res.Header), "websocket"; g != w {
		t.Fatalf("Upgrade header mismatch\n\tgot:  %q\n\twant: %q", g, w)
	}

	rwc, ok := res.Body.(io.ReadWriteCloser)
	if !ok {
		t.Fatalf("Response body type mismatch, got %T, want io.ReadWriteCloser", res.Body)
	}

	if got, want := res.Header.Get("X-Modified"), "true"; got != want {
		t.Errorf("response X-Modified header = %q; want %q", got, want)
	}

	if _, err := io.WriteString(rwc, "Hello\n"); err != nil {
		t.Fatalf("Failed to write first message: %v", err)
	}

	// Read loop.

	br := bufio.NewReader(rwc)
	for {
		line, err := br.ReadString('\n')
		switch {
		case line == terminalMsg: // this case before "err == io.EOF"
			t.Fatalf("The websocket request was not canceled, unfortunately!")

		case err == io.EOF:
			return

		case err != nil:
			t.Fatalf("Unexpected error: %v", err)

		case line == nthResponse(0): // We've gotten the first response back
			// Let's trigger a cancel.
			close(triggerCancelCh)
		}
	}
}

func TestUnannouncedTrailer(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		w.Header().Set(http.TrailerPrefix+"X-Unannounced-Trailer", "unannounced_trailer_value")
	}))
	defer backend.Close()
	backendURL, err := url.Parse(backend.URL)
	if err != nil {
		t.Fatal(err)
	}
	proxyHandler := NewSingleHostReverseProxy(backendURL)
	proxyHandler.ErrorLog = log.New(io.Discard, "", 0) // quiet for tests
	frontend := httptest.NewServer(proxyHandler)
	defer frontend.Close()
	frontendClient := frontend.Client()

	res, err := frontendClient.Get(frontend.URL)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}

	io.ReadAll(res.Body)

	if g, w := res.Trailer.Get("X-Unannounced-Trailer"), "unannounced_trailer_value"; g != w {
		t.Errorf("Trailer(X-Unannounced-Trailer) = %q; want %q", g, w)
	}

}

func TestSingleJoinSlash(t *testing.T) {
	tests := []struct {
		slasha   string
		slashb   string
		expected string
	}{
		{"https://www.google.com/", "/favicon.ico", "https://www.google.com/favicon.ico"},
		{"https://www.google.com", "/favicon.ico", "https://www.google.com/favicon.ico"},
		{"https://www.google.com", "favicon.ico", "https://www.google.com/favicon.ico"},
		{"https://www.google.com", "", "https://www.google.com/"},
		{"", "favicon.ico", "/favicon.ico"},
	}
	for _, tt := range tests {
		if got := singleJoiningSlash(tt.slasha, tt.slashb); got != tt.expected {
			t.Errorf("singleJoiningSlash(%q,%q) want %q got %q",
				tt.slasha,
				tt.slashb,
				tt.expected,
				got)
		}
	}
}

func TestJoinURLPath(t *testing.T) {
	tests := []struct {
		a        *url.URL
		b        *url.URL
		wantPath string
		wantRaw  string
	}{
		{&url.URL{Path: "/a/b"}, &url.URL{Path: "/c"}, "/a/b/c", ""},
		{&url.URL{Path: "/a/b", RawPath: "badpath"}, &url.URL{Path: "c"}, "/a/b/c", "/a/b/c"},
		{&url.URL{Path: "/a/b", RawPath: "/a%2Fb"}, &url.URL{Path: "/c"}, "/a/b/c", "/a%2Fb/c"},
		{&url.URL{Path: "/a/b", RawPath: "/a%2Fb"}, &url.URL{Path: "/c"}, "/a/b/c", "/a%2Fb/c"},
		{&url.URL{Path: "/a/b/", RawPath: "/a%2Fb%2F"}, &url.URL{Path: "c"}, "/a/b//c", "/a%2Fb%2F/c"},
		{&url.URL{Path: "/a/b/", RawPath: "/a%2Fb/"}, &url.URL{Path: "/c/d", RawPath: "/c%2Fd"}, "/a/b/c/d", "/a%2Fb/c%2Fd"},
	}

	for _, tt := range tests {
		p, rp := joinURLPath(tt.a, tt.b)
		if p != tt.wantPath || rp != tt.wantRaw {
			t.Errorf("joinURLPath(URL(%q,%q),URL(%q,%q)) want (%q,%q) got (%q,%q)",
				tt.a.Path, tt.a.RawPath,
				tt.b.Path, tt.b.RawPath,
				tt.wantPath, tt.wantRaw,
				p, rp)
		}
	}
}
