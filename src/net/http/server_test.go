// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Server unit tests

package http

import (
	"fmt"
	"testing"
	"time"
)

func TestServerTLSHandshakeTimeout(t *testing.T) {
	tests := []struct {
		s    *Server
		want time.Duration
	}{
		{
			s:    &Server{},
			want: 0,
		},
		{
			s: &Server{
				ReadTimeout: -1,
			},
			want: 0,
		},
		{
			s: &Server{
				ReadTimeout: 5 * time.Second,
			},
			want: 5 * time.Second,
		},
		{
			s: &Server{
				ReadTimeout:  5 * time.Second,
				WriteTimeout: -1,
			},
			want: 5 * time.Second,
		},
		{
			s: &Server{
				ReadTimeout:  5 * time.Second,
				WriteTimeout: 4 * time.Second,
			},
			want: 4 * time.Second,
		},
		{
			s: &Server{
				ReadTimeout:       5 * time.Second,
				ReadHeaderTimeout: 2 * time.Second,
				WriteTimeout:      4 * time.Second,
			},
			want: 2 * time.Second,
		},
	}
	for i, tt := range tests {
		got := tt.s.tlsHandshakeTimeout()
		if got != tt.want {
			t.Errorf("%d. got %v; want %v", i, got, tt.want)
		}
	}
}

func BenchmarkServerMatch(b *testing.B) {
	fn := func(w ResponseWriter, r *Request) {
		fmt.Fprintf(w, "OK")
	}
	mux := NewServeMux()
	mux.HandleFunc("/", fn)
	mux.HandleFunc("/index", fn)
	mux.HandleFunc("/home", fn)
	mux.HandleFunc("/about", fn)
	mux.HandleFunc("/contact", fn)
	mux.HandleFunc("/robots.txt", fn)
	mux.HandleFunc("/products/", fn)
	mux.HandleFunc("/products/1", fn)
	mux.HandleFunc("/products/2", fn)
	mux.HandleFunc("/products/3", fn)
	mux.HandleFunc("/products/3/image.jpg", fn)
	mux.HandleFunc("/admin", fn)
	mux.HandleFunc("/admin/products/", fn)
	mux.HandleFunc("/admin/products/create", fn)
	mux.HandleFunc("/admin/products/update", fn)
	mux.HandleFunc("/admin/products/delete", fn)

	paths := []string{"/", "/notfound", "/admin/", "/admin/foo", "/contact", "/products",
		"/products/", "/products/3/image.jpg"}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		if h, p := mux.match(paths[i%len(paths)]); h != nil && p == "" {
			b.Error("impossible")
		}
	}
	b.StopTimer()
}
