// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Server unit tests

package http

import (
	"fmt"
	"testing"
)

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
