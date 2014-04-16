// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httptest

import (
	"io/ioutil"
	"net/http"
	"testing"
	"time"
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

func TestIssue7264(t *testing.T) {
	for i := 0; i < 1000; i++ {
		func() {
			inHandler := make(chan bool, 1)
			ts := NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				inHandler <- true
			}))
			defer ts.Close()
			tr := &http.Transport{
				ResponseHeaderTimeout: time.Nanosecond,
			}
			defer tr.CloseIdleConnections()
			c := &http.Client{Transport: tr}
			res, err := c.Get(ts.URL)
			<-inHandler
			if err == nil {
				res.Body.Close()
			}
		}()
	}
}
