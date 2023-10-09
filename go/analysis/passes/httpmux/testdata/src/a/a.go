// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the httpmux checker.

package a

import "net/http"

func _() {
	http.HandleFunc("GET /x", nil)  // want "enhanced ServeMux pattern"
	http.HandleFunc("/{a}/b/", nil) // want "enhanced ServeMux pattern"
	mux := http.NewServeMux()
	mux.Handle("example.com/c/{d}", nil) // want "enhanced ServeMux pattern"
	mux.HandleFunc("/{x...}", nil)       // want "enhanced ServeMux pattern"

	// Should not match.

	// not an enhanced pattern
	http.Handle("/", nil)

	// invalid wildcard; will panic in 1.22
	http.HandleFunc("/{/a/}", nil)
	mux.Handle("/{1}", nil)
	mux.Handle("/x{a}", nil)

	// right package, wrong method
	http.ParseTime("GET /")

	// right function name, wrong package
	Handle("GET /", nil)
	HandleFunc("GET /", nil)

	// right method name, wrong type
	var sm ServeMux
	sm.Handle("example.com/c/{d}", nil)
	sm.HandleFunc("method /{x...}", nil)
}

func Handle(pat string, x any)     {}
func HandleFunc(pat string, x any) {}

type ServeMux struct{}

func (*ServeMux) Handle(pat string, x any)     {}
func (*ServeMux) HandleFunc(pat string, x any) {}
