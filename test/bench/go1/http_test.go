// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package go1

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// BenchmarkHTTPClientServer benchmarks both the HTTP client and the HTTP server,
// on small requests.
func BenchmarkHTTPClientServer(b *testing.B) {
	msg := []byte("Hello world.\n")
	ts := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		rw.Write(msg)
	}))
	defer ts.Close()

	tr := &http.Transport{}
	defer tr.CloseIdleConnections()
	cl := &http.Client{
		Transport: tr,
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		res, err := cl.Get(ts.URL)
		if err != nil {
			b.Fatal("Get:", err)
		}
		all, err := io.ReadAll(res.Body)
		if err != nil {
			b.Fatal("ReadAll:", err)
		}
		if !bytes.Equal(all, msg) {
			b.Fatalf("Got body %q; want %q", all, msg)
		}
	}
}
