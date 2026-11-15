// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include <stdlib.h>

extern int goCallback(int v);

static int target(int *p, int doCall) {
	int v = *p;
	if (doCall) {
		v = goCallback(v);
	}
	return v + 1;
}

#cgo noescape target
#cgo nocallback target
*/
import "C"

import (
	"fmt"
	"os"
	"runtime/debug"
	"strings"
)

//export goCallback
func goCallback(v C.int) C.int { return v + 1 }

//go:noinline
func forbidden(v C.int, want string) (panicked bool, bad string) {
	defer func() {
		if r := recover(); r != nil {
			if strings.Contains(fmt.Sprint(r), want) {
				panicked = true
			} else {
				bad = fmt.Sprint("wrong panic class: ", r)
			}
		}
	}()
	got := C.target(&v, 1)
	bad = fmt.Sprintf("returned %d instead of panicking", int(got))
	return false, bad
}

func repeatWant() string {
	bi, ok := debug.ReadBuildInfo()
	if !ok {
		return "unexpected setting cgoNoCallback"
	}
	for _, kv := range bi.Settings {
		if kv.Key != "GOEXPERIMENT" {
			continue
		}
		for name := range strings.SplitSeq(kv.Value, ",") {
			if name == "logosfastcgo" {
				return "called back into Go"
			}
		}
	}
	return "unexpected setting cgoNoCallback"
}

func worker(id int, bad chan<- string) {
	wantRepeat := repeatWant()
	for n := range 100 {
		want := wantRepeat
		if n == 0 {
			want = "called back into Go"
		}
		ok, msg := forbidden(C.int(41+n), want)
		if msg != "" {
			bad <- fmt.Sprintf("goroutine %d iter %d %s", id, n, msg)
			return
		}
		if !ok {
			bad <- fmt.Sprintf("goroutine %d iter %d did not panic", id, n)
			return
		}
	}
	bad <- ""
	select {}
}

func run() (err string) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Sprint("unexpected panic: ", r)
		}
	}()
	bad := make(chan string, 8)
	for id := range 8 {
		go worker(id, bad)
	}
	for range 8 {
		if s := <-bad; s != "" {
			return s
		}
	}
	return ""
}

func main() {
	if err := run(); err != "" {
		fmt.Println("LAW cgo both forbid par", err)
		os.Exit(1)
	}
	fmt.Println("ok cgo both forbid par")
}
