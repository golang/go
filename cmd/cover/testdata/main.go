// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test runner for coverage test. This file is not coverage-annotated; test.go is.
// It knows the coverage counter is called "coverTest".

package main

import (
	"fmt"
	"os"
)

func main() {
	testAll()
	verify()
}

type block struct {
	count uint32
	line  uint32
}

var counters = make(map[block]bool)

// check records the location and expected value for a counter.
func check(line, count uint32) {
	b := block{
		count,
		line,
	}
	counters[b] = true
}

// verify checks the expected counts against the actual. It runs after the test has completed.
func verify() {
	ok := true
	for b := range counters {
		got := count(b.line)
		if b.count == anything && got != 0 {
			got = anything
		}
		if got != b.count {
			fmt.Fprintf(os.Stderr, "test_go:%d expected count %d got %d\n", b.line, b.count, got)
			ok = false
		}
	}
	if !ok {
		fmt.Fprintf(os.Stderr, "FAIL\n")
		os.Exit(2)
	}
}

func count(line uint32) uint32 {
	// Linear search is fine.
	for i := range coverTest.Count {
		lo, hi := coverTest.Pos[3*i], coverTest.Pos[3*i+1]
		if lo <= line && line <= hi {
			return coverTest.Count[i]
		}
	}
	return 0
}
