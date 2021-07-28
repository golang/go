// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

// Index returns the index of x in s, or -1 if not found.
func Index[T comparable](s []T, x T) int {
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

type obj2 struct {
	x int8
	y float64
}

type obj3 struct {
	x int64
	y int8
}

type inner struct {
	y int64
	z int32
}

type obj4 struct {
	x int32
	s inner
}

func main() {
	want := 2

	vec1 := []string{"ab", "cd", "ef"}
	if got := Index(vec1, "ef"); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	vec2 := []byte{'c', '6', '@'}
	if got := Index(vec2, '@'); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	vec3 := []*obj{&obj{2}, &obj{42}, &obj{1}}
	if got := Index(vec3, vec3[2]); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	vec4 := []obj2{obj2{2, 3.0}, obj2{3, 4.0}, obj2{4, 5.0}}
	if got := Index(vec4, vec4[2]); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	vec5 := []obj3{obj3{2, 3}, obj3{3, 4}, obj3{4, 5}}
	if got := Index(vec5, vec5[2]); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	vec6 := []obj4{obj4{2, inner{3, 4}}, obj4{3, inner{4, 5}}, obj4{4, inner{5, 6}}}
	if got := Index(vec6, vec6[2]); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}
}
