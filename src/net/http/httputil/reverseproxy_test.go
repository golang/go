// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reverse proxy tests.

package httputil

import (
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

const fakeHopHeader = "X-Fake-Hop-Header-For-Test"

func init() {
	hopHeaders = append(hopHeaders, fakeHopHeader)
}

func TestReverseProxy(t *testing.T) {
	const backendResponse = "I am the backend"
	const backendStatus = 404
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if len(r.TransferEncoding) > 0 {
			t.Errorf("backend got unexpected TransferEncoding: %v", r.TransferEncoding)
		}
		if r.Header.Get("X-Forwarded-For") == "" {
			t.Errorf("didn't get X-Forwarded-For header")
		}
		if c := r.Header.Get("Connection"); c != "" {
			t.Errorf("handler got Connection header value %q", c)
		}
		if c := r.Header.Get("Upgrade"); c != "" {
			t.Errorf("handler got Upgrade header value %q", c)
		}
		if g, e := r.Host, "some-name"; g != e {
			t.Errorf("backend got Host header %q, want %q", g, e)
		}
		w.Header().Set("X-Foo", "bar")
		w.Header().Set("Upgrade", "foo")
		w.Header().Set(fakeHopHeader, "foo")
		w.Header().Add("X-Multi-Value", "foo")
		w.Header().Add("X-Multi-Value", "bar")
		http.SetCookie(w, &http.Cookie{Name: "flavor", Value: "chocolateChip"})
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
	getReq.Header.Set("Upgrade", "foo")
	getReq.Close = true
	res, err := http.DefaultClient.Do(getReq)
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
	if g, e := len(res.Header["X-Multi-Value"]), 2; g != e {
		t.Errorf("got %d X-Multi-Value header values; expected %d", g, e)
	}
	if g, e := len(res.Header["Set-Cookie"]), 1; g != e {
		t.Fatalf("got %d SetCookies, want %d", g, e)
	}
	if cookie := res.Cookies()[0]; cookie.Name != "flavor" {
		t.Errorf("unexpected cookie %q", cookie.Name)
	}
	bodyBytes, _ := ioutil.ReadAll(res.Body)
	if g, e := string(bodyBytes), backendResponse; g != e {
		t.Errorf("got body %q; expected %q", g, e)
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
	res, err := http.DefaultClient.Do(getReq)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if g, e := res.StatusCode, backendStatus; g != e {
		t.Errorf("got res.StatusCode %d; expected %d", g, e)
	}
	bodyBytes, _ := ioutil.ReadAll(res.Body)
	if g, e := string(bodyBytes), backendResponse; g != e {
		t.Errorf("got body %q; expected %q", g, e)
	}
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
		res, err := http.DefaultClient.Do(req)
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

	done := make(chan bool)
	onExitFlushLoop = func() { done <- true }
	defer func() { onExitFlushLoop = nil }()

	frontend := httptest.NewServer(proxyHandler)
	defer frontend.Close()

	req, _ := http.NewRequest("GET", frontend.URL, nil)
	req.Close = true
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	defer res.Body.Close()
	if bodyBytes, _ := ioutil.ReadAll(res.Body); string(bodyBytes) != expected {
		t.Errorf("got body %q; expected %q", bodyBytes, expected)
	}

	select {
	case <-done:
		// OK
	case <-time.After(5 * time.Second):
		t.Error("maxLatencyWriter flushLoop() never exited")
	}
}
