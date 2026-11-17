// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include <stdlib.h>

static int target(int *p) {
	return *p + 1;
}

#cgo nocallback target
*/
import "C"

import (
	"fmt"
	"os"
)

//go:noinline
func call(v C.int) C.int {
	return C.target(&v)
}

func worker(id int, bad chan<- string) {
	for n := range 100 {
		v := C.int(41 + n)
		if got := call(v); got != v+1 {
			bad <- fmt.Sprintf("goroutine %d iter %d got %d want %d", id, n, int(got), int(v+1))
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
		fmt.Println("LAW cgo nocallback plain par", err)
		os.Exit(1)
	}
	fmt.Println("ok cgo nocallback plain par")
}
