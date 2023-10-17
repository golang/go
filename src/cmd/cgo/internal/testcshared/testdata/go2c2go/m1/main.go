// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// extern int CFunc(void);
import "C"

import (
	"fmt"
	"os"
)

func main() {
	got := C.CFunc()
	const want = (1 << 8) | 2
	if got != want {
		fmt.Printf("got %#x, want %#x\n", got, want)
		os.Exit(1)
	}
}
