// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"io"
	"log"
	"os"
)

func main() {
	f, err := os.OpenFile(os.Getenv("TEST_OUTPUT"), os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		log.Fatalf("os.Open failed: %s", err)
	}
	defer f.Close()
	b, err := io.ReadAll(os.Stdin)
	if err != nil {
		log.Fatalf("io.ReadAll(os.Stdin) failed: %s", err)
	}
	if len(b) != 0 {
		log.Fatalf("io.ReadAll(os.Stdin) returned non-nil: %x", b)
	}
	fmt.Fprintf(os.Stdout, "stdout\n")
	fmt.Fprintf(os.Stderr, "stderr\n")
}
