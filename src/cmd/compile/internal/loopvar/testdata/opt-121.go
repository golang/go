// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.21

package main

import (
	"fmt"
	"os"
)

var is []func() int

func inline(j, k int) []*int {
	var a []*int
	for private := j; private < k; private++ {
		a = append(a, &private)
	}
	return a
}

//go:noinline
func notinline(j, k int) ([]*int, *int) {
	for shared := j; shared < k; shared++ {
		if shared == k/2 {
			// want the call inlined, want "private" in that inline to be transformed,
			// (believe it ends up on init node of the return).
			// but do not want "shared" transformed,
			return inline(j, k), &shared
		}
	}
	return nil, &j
}

func main() {
	a, p := notinline(2, 9)
	fmt.Printf("a[0]=%d,*p=%d\n", *a[0], *p)
	if *a[0] != 2 {
		os.Exit(1)
	}
}
