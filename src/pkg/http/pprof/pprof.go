// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package pprof serves via its HTTP server runtime profiling data
// in the format expected by the pprof visualization tool.
// For more information about pprof, see
// http://code.google.com/p/google-perftools/.
//
// The package is typically only imported for the side effect of
// registering its HTTP handlers.
// The handled paths all begin with /debug/pprof/.
//
// To use pprof, link this package into your program:
//	import _ "http/pprof"
//
// Then use the pprof tool to look at the heap profile:
//
//	pprof http://localhost:6060/debug/pprof/heap
//
package pprof

import (
	"bufio"
	"fmt"
	"http"
	"os"
	"runtime"
	"runtime/pprof"
	"strconv"
	"strings"
)

func init() {
	http.Handle("/debug/pprof/cmdline", http.HandlerFunc(Cmdline))
	http.Handle("/debug/pprof/heap", http.HandlerFunc(Heap))
	http.Handle("/debug/pprof/symbol", http.HandlerFunc(Symbol))
}

// Cmdline responds with the running program's
// command line, with arguments separated by NUL bytes.
// The package initialization registers it as /debug/pprof/cmdline.
func Cmdline(w http.ResponseWriter, r *http.Request) {
	w.SetHeader("content-type", "text/plain; charset=utf-8")
	fmt.Fprintf(w, strings.Join(os.Args, "\x00"))
}

// Heap responds with the pprof-formatted heap profile.
// The package initialization registers it as /debug/pprof/heap.
func Heap(w http.ResponseWriter, r *http.Request) {
	w.SetHeader("content-type", "text/plain; charset=utf-8")
	pprof.WriteHeapProfile(w)
}

// Symbol looks up the program counters listed in the request,
// responding with a table mapping program counters to function names.
// The package initialization registers it as /debug/pprof/symbol.
func Symbol(w http.ResponseWriter, r *http.Request) {
	w.SetHeader("content-type", "text/plain; charset=utf-8")

	// We don't know how many symbols we have, but we
	// do have symbol information.  Pprof only cares whether
	// this number is 0 (no symbols available) or > 0.
	fmt.Fprintf(w, "num_symbols: 1\n")

	var b *bufio.Reader
	if r.Method == "POST" {
		b = bufio.NewReader(r.Body)
	} else {
		b = bufio.NewReader(strings.NewReader(r.URL.RawQuery))
	}

	for {
		word, err := b.ReadSlice('+')
		if err == nil {
			word = word[0 : len(word)-1] // trim +
		}
		pc, _ := strconv.Btoui64(string(word), 0)
		if pc != 0 {
			f := runtime.FuncForPC(uintptr(pc))
			if f != nil {
				fmt.Fprintf(w, "%#x %s\n", pc, f.Name())
			}
		}

		// Wait until here to check for err; the last
		// symbol will have an err because it doesn't end in +.
		if err != nil {
			break
		}
	}
}
