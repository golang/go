// run

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Disable mapping annotations, which only exists for Linux.

//go:debug decoratemappings=0
//go:build linux

package main

import (
	"log"
	"os"
	"strings"
)

func main() {
	b, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		log.Fatalf("Error reading: %v", err)
	}

	if strings.Contains(string(b), "[anon: Go:") {
		log.Printf("/proc/self/maps:\n%s", string(b))
		log.Fatalf("/proc/self/maps contains Go annotation")
	}
}
