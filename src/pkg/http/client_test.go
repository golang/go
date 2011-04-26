// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for client.go

package http_test

import (
	"fmt"
	. "http"
	"http/httptest"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
	"testing"
)

var robotsTxtHandler = HandlerFunc(func(w ResponseWriter, r *Request) {
	w.Header().Set("Last-Modified", "sometime")
	fmt.Fprintf(w, "User-agent: go\nDisallow: /something/")
})

func TestClient(t *testing.T) {
	ts := httptest.NewServer(robotsTxtHandler)
	defer ts.Close()

	r, _, err := Get(ts.URL)
	var b []byte
	if err == nil {
		b, err = ioutil.ReadAll(r.Body)
		r.Body.Close()
	}
	if err != nil {
		t.Error(err)
	} else if s := string(b); !strings.HasPrefix(s, "User-agent:") {
		t.Errorf("Incorrect page body (did not begin with User-agent): %q", s)
	}
}

func TestClientHead(t *testing.T) {
	ts := httptest.NewServer(robotsTxtHandler)
	defer ts.Close()

	r, err := Head(ts.URL)
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

func (t *recordingTransport) RoundTrip(req *Request) (resp *Response, err os.Error) {
	t.req = req
	return nil, os.NewError("dummy impl")
}

func TestGetRequestFormat(t *testing.T) {
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

func TestRedirects(t *testing.T) {
	var ts *httptest.Server
	ts = httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		n, _ := strconv.Atoi(r.FormValue("n"))
		// Test Referer header. (7 is arbitrary position to test at)
		if n == 7 {
			if g, e := r.Referer, ts.URL+"/?n=6"; e != g {
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
	_, _, err := c.Get(ts.URL)
	if e, g := "Get /?n=10: stopped after 10 redirects", fmt.Sprintf("%v", err); e != g {
		t.Errorf("with default client, expected error %q, got %q", e, g)
	}

	var checkErr os.Error
	var lastVia []*Request
	c = &Client{CheckRedirect: func(_ *Request, via []*Request) os.Error {
		lastVia = via
		return checkErr
	}}
	_, finalUrl, err := c.Get(ts.URL)
	if e, g := "<nil>", fmt.Sprintf("%v", err); e != g {
		t.Errorf("with custom client, expected error %q, got %q", e, g)
	}
	if !strings.HasSuffix(finalUrl, "/?n=15") {
		t.Errorf("expected final url to end in /?n=15; got url %q", finalUrl)
	}
	if e, g := 15, len(lastVia); e != g {
		t.Errorf("expected lastVia to have contained %d elements; got %d", e, g)
	}

	checkErr = os.NewError("no redirects allowed")
	_, finalUrl, err = c.Get(ts.URL)
	if e, g := "Get /?n=1: no redirects allowed", fmt.Sprintf("%v", err); e != g {
		t.Errorf("with redirects forbidden, expected error %q, got %q", e, g)
	}
}
