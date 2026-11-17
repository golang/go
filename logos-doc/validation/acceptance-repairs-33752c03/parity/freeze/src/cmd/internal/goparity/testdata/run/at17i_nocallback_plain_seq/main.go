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

func safemain() string {
	if got := call(41); got != 42 {
		return fmt.Sprintf("first call got %d want 42", int(got))
	}
	if got := call(41); got != 42 {
		return fmt.Sprintf("second call got %d want 42", int(got))
	}
	if got := call(41); got != 42 {
		return fmt.Sprintf("third call got %d want 42", int(got))
	}
	return ""
}

func main() {
	if err := safemain(); err != "" {
		fmt.Println("LAW cgo nocallback plain seq", err)
		os.Exit(1)
	}
	fmt.Println("ok cgo nocallback plain seq")
}
