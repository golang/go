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

// checkVal is a version of check that returns its extra argument,
// so it can be used in conditionals.
func checkVal(line, count uint32, val int) int {
	b := block{
		count,
		line,
	}
	counters[b] = true
	return val
}

var PASS = true

// verify checks the expected counts against the actual. It runs after the test has completed.
func verify() {
	for b := range counters {
		got, index := count(b.line)
		if b.count == anything && got != 0 {
			got = anything
		}
		if got != b.count {
			fmt.Fprintf(os.Stderr, "test_go:%d expected count %d got %d [counter %d]\n", b.line, b.count, got, index)
			PASS = false
		}
	}
	if !PASS {
		fmt.Fprintf(os.Stderr, "FAIL\n")
		os.Exit(2)
	}
}

// count returns the count and index for the counter at the specified line.
func count(line uint32) (uint32, int) {
	// Linear search is fine. Choose perfect fit over approximate.
	// We can have a closing brace for a range on the same line as a condition for an "else if"
	// and we don't want that brace to steal the count for the condition on the "if".
	// Therefore we test for a perfect (lo==line && hi==line) match, but if we can't
	// find that we take the first imperfect match.
	index := -1
	indexLo := uint32(1e9)
	for i := range coverTest.Count {
		lo, hi := coverTest.Pos[3*i], coverTest.Pos[3*i+1]
		if lo == line && line == hi {
			return coverTest.Count[i], i
		}
		// Choose the earliest match (the counters are in unpredictable order).
		if lo <= line && line <= hi && indexLo > lo {
			index = i
			indexLo = lo
		}
	}
	if index == -1 {
		fmt.Fprintln(os.Stderr, "cover_test: no counter for line", line)
		PASS = false
		return 0, 0
	}
	return coverTest.Count[index], index
}
