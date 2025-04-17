// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"log"
	"os"
)

func main() {
	if os.Geteuid() == os.Getuid() {
		os.Exit(99)
	}

	fmt.Fprintf(os.Stdout, "GOTRACEBACK=%s\n", os.Getenv("GOTRACEBACK"))
	f, err := os.OpenFile(os.Getenv("TEST_OUTPUT"), os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		log.Fatalf("os.Open failed: %s", err)
	}
	defer f.Close()
	fmt.Fprintf(os.Stderr, "hello\n")
}
