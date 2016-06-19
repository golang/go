// build

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file tests that required algs are generated,
// even when similar types have been marked elsewhere
// as not needing algs. See CLs 19769 and 19770.

package main

import "fmt"

//go:noinline
func f(m map[[8]string]int) int {
	var k [8]string
	return m[k]
}

//go:noinline
func g(m map[[8]interface{}]int) int {
	var k [8]interface{}
	return m[k]
}

//go:noinline
func h(m map[[2]string]int) int {
	var k [2]string
	return m[k]
}

type T map[string]interface{}

func v(x ...string) string {
	return x[0] + x[1]
}

func main() {
	fmt.Println(
		f(map[[8]string]int{}),
		g(map[[8]interface{}]int{}),
		h(map[[2]string]int{}),
		v("a", "b"),
	)
}
