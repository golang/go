// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

func f(a []byte) ([]byte, []byte) {
	return a, []byte("abc")
}

func g(a []byte) ([]byte, string) {
	return a, "abc"
}

func h(m map[int]int) (map[int]int, int) {
	return m, 0
}

func main() {
	a := []byte{1, 2, 3}
	n := copy(f(a))
	fmt.Println(n, a)

	b := []byte{1, 2, 3}
	n = copy(g(b))
	fmt.Println(n, b)

	m := map[int]int{0: 0}
	fmt.Println(len(m))
	delete(h(m))
	fmt.Println(len(m))
}
