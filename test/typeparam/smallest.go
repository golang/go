// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

type Ordered interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64 |
		~string
}

func Smallest[T Ordered](s []T) T {
	r := s[0] // panics if slice is empty
	for _, v := range s[1:] {
		if v < r {
			r = v
		}
	}
	return r
}

func main() {
	vec1 := []float64{5.3, 1.2, 32.8}
	vec2 := []string{"abc", "def", "aaa"}

	want1 := 1.2
	if got := Smallest(vec1); got != want1 {
		panic(fmt.Sprintf("got %d, want %d", got, want1))
	}
	want2 := "aaa"
	if got := Smallest(vec2); got != want2 {
		panic(fmt.Sprintf("got %d, want %d", got, want2))
	}
}
