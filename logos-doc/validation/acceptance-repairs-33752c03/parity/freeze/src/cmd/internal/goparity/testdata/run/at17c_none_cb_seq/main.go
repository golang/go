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

func safemain() string {
	for i := range 3 {
		v := C.int(41 + i)
		if got := call(v, 1); got != v+2 {
			return fmt.Sprintf("call %d got %d want %d", i, int(got), int(v+2))
		}
	}
	return ""
}

func main() {
	if err := safemain(); err != "" {
		fmt.Println("LAW cgo none cb seq", err)
		os.Exit(1)
	}
	fmt.Println("ok cgo none cb seq")
}
