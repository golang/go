// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for transport.go

package http_test

import (
	"fmt"
	. "http"
	"http/httptest"
	"io/ioutil"
	"os"
	"testing"
	"time"
)

// TODO: test 5 pipelined requests with responses: 1) OK, 2) OK, Connection: Close
//       and then verify that the final 2 responses get errors back.

// hostPortHandler writes back the client's "host:port".
var hostPortHandler = HandlerFunc(func(w ResponseWriter, r *Request) {
	if r.FormValue("close") == "true" {
		w.Header().Set("Connection", "close")
	}
	fmt.Fprintf(w, "%s", r.RemoteAddr)
})

// Two subsequent requests and verify their response is the same.
// The response from the server is our own IP:port
func TestTransportKeepAlives(t *testing.T) {
	ts := httptest.NewServer(hostPortHandler)
	defer ts.Close()

	for _, disableKeepAlive := range []bool{false, true} {
		tr := &Transport{DisableKeepAlives: disableKeepAlive}
		c := &Client{Transport: tr}

		fetch := func(n int) string {
			res, _, err := c.Get(ts.URL)
			if err != nil {
				t.Fatalf("error in disableKeepAlive=%v, req #%d, GET: %v", disableKeepAlive, n, err)
			}
			body, err := ioutil.ReadAll(res.Body)
			if err != nil {
				t.Fatalf("error in disableKeepAlive=%v, req #%d, ReadAll: %v", disableKeepAlive, n, err)
			}
			return string(body)
		}

		body1 := fetch(1)
		body2 := fetch(2)

		bodiesDiffer := body1 != body2
		if bodiesDiffer != disableKeepAlive {
			t.Errorf("error in disableKeepAlive=%v. unexpected bodiesDiffer=%v; body1=%q; body2=%q",
				disableKeepAlive, bodiesDiffer, body1, body2)
		}
	}
}

func TestTransportConnectionCloseOnResponse(t *testing.T) {
	ts := httptest.NewServer(hostPortHandler)
	defer ts.Close()

	for _, connectionClose := range []bool{false, true} {
		tr := &Transport{}
		c := &Client{Transport: tr}

		fetch := func(n int) string {
			req := new(Request)
			var err os.Error
			req.URL, err = ParseURL(ts.URL + fmt.Sprintf("?close=%v", connectionClose))
			if err != nil {
				t.Fatalf("URL parse error: %v", err)
			}
			req.Method = "GET"
			req.Proto = "HTTP/1.1"
			req.ProtoMajor = 1
			req.ProtoMinor = 1

			res, err := c.Do(req)
			if err != nil {
				t.Fatalf("error in connectionClose=%v, req #%d, Do: %v", connectionClose, n, err)
			}
			body, err := ioutil.ReadAll(res.Body)
			if err != nil {
				t.Fatalf("error in connectionClose=%v, req #%d, ReadAll: %v", connectionClose, n, err)
			}
			return string(body)
		}

		body1 := fetch(1)
		body2 := fetch(2)
		bodiesDiffer := body1 != body2
		if bodiesDiffer != connectionClose {
			t.Errorf("error in connectionClose=%v. unexpected bodiesDiffer=%v; body1=%q; body2=%q",
				connectionClose, bodiesDiffer, body1, body2)
		}
	}
}

func TestTransportConnectionCloseOnRequest(t *testing.T) {
	ts := httptest.NewServer(hostPortHandler)
	defer ts.Close()

	for _, connectionClose := range []bool{false, true} {
		tr := &Transport{}
		c := &Client{Transport: tr}

		fetch := func(n int) string {
			req := new(Request)
			var err os.Error
			req.URL, err = ParseURL(ts.URL)
			if err != nil {
				t.Fatalf("URL parse error: %v", err)
			}
			req.Method = "GET"
			req.Proto = "HTTP/1.1"
			req.ProtoMajor = 1
			req.ProtoMinor = 1
			req.Close = connectionClose

			res, err := c.Do(req)
			if err != nil {
				t.Fatalf("error in connectionClose=%v, req #%d, Do: %v", connectionClose, n, err)
			}
			body, err := ioutil.ReadAll(res.Body)
			if err != nil {
				t.Fatalf("error in connectionClose=%v, req #%d, ReadAll: %v", connectionClose, n, err)
			}
			return string(body)
		}

		body1 := fetch(1)
		body2 := fetch(2)
		bodiesDiffer := body1 != body2
		if bodiesDiffer != connectionClose {
			t.Errorf("error in connectionClose=%v. unexpected bodiesDiffer=%v; body1=%q; body2=%q",
				connectionClose, bodiesDiffer, body1, body2)
		}
	}
}

func TestTransportIdleCacheKeys(t *testing.T) {
	ts := httptest.NewServer(hostPortHandler)
	defer ts.Close()

	tr := &Transport{DisableKeepAlives: false}
	c := &Client{Transport: tr}

	if e, g := 0, len(tr.IdleConnKeysForTesting()); e != g {
		t.Errorf("After CloseIdleConnections expected %d idle conn cache keys; got %d", e, g)
	}

	if _, _, err := c.Get(ts.URL); err != nil {
		t.Error(err)
	}

	keys := tr.IdleConnKeysForTesting()
	if e, g := 1, len(keys); e != g {
		t.Fatalf("After Get expected %d idle conn cache keys; got %d", e, g)
	}

	if e := "|http|" + ts.Listener.Addr().String(); keys[0] != e {
		t.Errorf("Expected idle cache key %q; got %q", e, keys[0])
	}

	tr.CloseIdleConnections()
	if e, g := 0, len(tr.IdleConnKeysForTesting()); e != g {
		t.Errorf("After CloseIdleConnections expected %d idle conn cache keys; got %d", e, g)
	}
}

func TestTransportMaxPerHostIdleConns(t *testing.T) {
	ch := make(chan string)
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "%s", <-ch)
	}))
	defer ts.Close()
	maxIdleConns := 2
	tr := &Transport{DisableKeepAlives: false, MaxIdleConnsPerHost: maxIdleConns}
	c := &Client{Transport: tr}

	// Start 3 outstanding requests (will hang until we write to
	// ch)
	donech := make(chan bool)
	doReq := func() {
		c.Get(ts.URL)
		donech <- true
	}
	go doReq()
	go doReq()
	go doReq()

	if e, g := 0, len(tr.IdleConnKeysForTesting()); e != g {
		t.Fatalf("Before writes, expected %d idle conn cache keys; got %d", e, g)
	}

	ch <- "res1"
	<-donech
	keys := tr.IdleConnKeysForTesting()
	if e, g := 1, len(keys); e != g {
		t.Fatalf("after first response, expected %d idle conn cache keys; got %d", e, g)
	}
	cacheKey := "|http|" + ts.Listener.Addr().String()
	if keys[0] != cacheKey {
		t.Fatalf("Expected idle cache key %q; got %q", cacheKey, keys[0])
	}
	if e, g := 1, tr.IdleConnCountForTesting(cacheKey); e != g {
		t.Errorf("after first response, expected %d idle conns; got %d", e, g)
	}

	ch <- "res2"
	<-donech
	if e, g := 2, tr.IdleConnCountForTesting(cacheKey); e != g {
		t.Errorf("after second response, expected %d idle conns; got %d", e, g)
	}

	ch <- "res3"
	<-donech
	if e, g := maxIdleConns, tr.IdleConnCountForTesting(cacheKey); e != g {
		t.Errorf("after third response, still expected %d idle conns; got %d", e, g)
	}
}

func TestTransportServerClosingUnexpectedly(t *testing.T) {
	ts := httptest.NewServer(hostPortHandler)
	defer ts.Close()

	tr := &Transport{}
	c := &Client{Transport: tr}

	fetch := func(n int) string {
		res, _, err := c.Get(ts.URL)
		if err != nil {
			t.Fatalf("error in req #%d, GET: %v", n, err)
		}
		body, err := ioutil.ReadAll(res.Body)
		if err != nil {
			t.Fatalf("error in req #%d, ReadAll: %v", n, err)
		}
		res.Body.Close()
		return string(body)
	}

	body1 := fetch(1)
	body2 := fetch(2)

	ts.CloseClientConnections() // surprise!
	time.Sleep(25e6)            // idle for a bit (test is inherently racey, but expectedly)

	body3 := fetch(3)

	if body1 != body2 {
		t.Errorf("expected body1 and body2 to be equal")
	}
	if body2 == body3 {
		t.Errorf("expected body2 and body3 to be different")
	}
}

func TestTransportNilURL(t *testing.T) {
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "Hi")
	}))
	defer ts.Close()

	req := new(Request)
	req.URL = nil // what we're actually testing
	req.Method = "GET"
	req.RawURL = ts.URL
	req.Proto = "HTTP/1.1"
	req.ProtoMajor = 1
	req.ProtoMinor = 1

	tr := &Transport{}
	res, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatalf("unexpected RoundTrip error: %v", err)
	}
	body, err := ioutil.ReadAll(res.Body)
	if g, e := string(body), "Hi"; g != e {
		t.Fatalf("Expected response body of %q; got %q", e, g)
	}
}
