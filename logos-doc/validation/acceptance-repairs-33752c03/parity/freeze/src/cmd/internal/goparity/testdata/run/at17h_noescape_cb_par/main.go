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
*/
import "C"

import (
	"fmt"
	"os"
)

//export goCallback
func goCallback(v C.int) C.int { return v + 1 }

//go:noinline
func call(v C.int, doCall C.int) C.int {
	return C.target(&v, doCall)
}

func worker(id int, bad chan<- string) {
	for n := range 100 {
		v := C.int(41 + n)
		if got := call(v, 1); got != v+2 {
			bad <- fmt.Sprintf("goroutine %d iter %d got %d want %d", id, n, int(got), int(v+2))
			return
		}
	}
	bad <- ""
}

func run() string {
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
		fmt.Println("LAW cgo noescape cb par", err)
		os.Exit(1)
	}
	fmt.Println("ok cgo noescape cb par")
}
