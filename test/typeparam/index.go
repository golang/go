// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

// Index returns the index of x in s, or -1 if not found.
func index[T comparable](s []T, x T) int {
	for i, v := range s {
		// v and x are type T, which has the comparable
		// constraint, so we can use == here.
		if v == x {
			return i
		}
	}
	return -1
}

type obj struct {
	x int
}

func main() {
	want := 2

	vec1 := []string{"ab", "cd", "ef"}
	if got := index(vec1, "ef"); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	vec2 := []byte{'c', '6', '@'}
	if got := index(vec2, '@'); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	vec3 := []*obj{&obj{2}, &obj{42}, &obj{1}}
	if got := index(vec3, vec3[2]); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}
}
