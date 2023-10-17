// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

func main() {
	x := [][]int{{1}}
	y := [][]int{{2, 3}}
	IntersectSS(x, y)
}

type list[E any] interface {
	~[]E
	Equal(x, y E) bool
}

// ss is a set of sets
type ss[E comparable, T []E] []T

func (ss[E, T]) Equal(a, b T) bool {
	return SetEq(a, b)
}

func IntersectSS[E comparable](x, y [][]E) [][]E {
	return IntersectT[[]E, ss[E, []E]](ss[E, []E](x), ss[E, []E](y))
}

func IntersectT[E any, L list[E]](x, y L) L {
	var z L
outer:
	for _, xe := range x {
		fmt.Println("xe", xe)
		for _, ye := range y {
			fmt.Println("ye", ye)
			fmt.Println("x", x)
			if x.Equal(xe, ye) {
				fmt.Println("appending")
				z = append(z, xe)
				continue outer
			}
		}
	}
	return z
}

func SetEq[S []E, E comparable](x, y S) bool {
	fmt.Println("SetEq", x, y)
outer:
	for _, xe := range x {
		for _, ye := range y {
			if xe == ye {
				continue outer
			}
		}
		return false // xs wasn't found in y
	}
	return true
}
