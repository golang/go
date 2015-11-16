// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httptest

import (
	"bufio"
	"io/ioutil"
	"net"
	"net/http"
	"testing"
)

func TestServer(t *testing.T) {
	ts := NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("hello"))
	}))
	defer ts.Close()
	res, err := http.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "hello" {
		t.Errorf("got %q, want hello", string(got))
	}
}

// Issue 12781
func TestGetAfterClose(t *testing.T) {
	ts := NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("hello"))
	}))

	res, err := http.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "hello" {
		t.Fatalf("got %q, want hello", string(got))
	}

	ts.Close()

	res, err = http.Get(ts.URL)
	if err == nil {
		body, _ := ioutil.ReadAll(res.Body)
		t.Fatalf("Unexected response after close: %v, %v, %s", res.Status, res.Header, body)
	}
}

func TestServerCloseBlocking(t *testing.T) {
	ts := NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("hello"))
	}))
	dial := func() net.Conn {
		c, err := net.Dial("tcp", ts.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		return c
	}

	// Keep one connection in StateNew (connected, but not sending anything)
	cnew := dial()
	defer cnew.Close()

	// Keep one connection in StateIdle (idle after a request)
	cidle := dial()
	defer cidle.Close()
	cidle.Write([]byte("HEAD / HTTP/1.1\r\nHost: foo\r\n\r\n"))
	_, err := http.ReadResponse(bufio.NewReader(cidle), nil)
	if err != nil {
		t.Fatal(err)
	}

	ts.Close() // test we don't hang here forever.
}
