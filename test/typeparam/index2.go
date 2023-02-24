// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Testing various generic uses of indexing, both for reads and writes.

package main

import "fmt"

// Can index an argument (read/write) constrained to be a slice or an array.
func Index1[T interface{ []int64 | [5]int64 }](x T) int64 {
	x[2] = 5
	return x[3]
}

// Can index an argument (read) constrained to be a byte array or a string.
func Index2[T interface{ []byte | string }](x T) byte {
	return x[3]
}

// Can index an argument (write) constrained to be a byte array, but not a string.
func Index2a[T interface{ []byte }](x T) byte {
	x[2] = 'b'
	return x[3]
}

// Can index an argument (read/write) constrained to be a map. Maps can't
// be combined with any other type for indexing purposes.
func Index3[T interface{ map[int]int64 }](x T) int64 {
	x[2] = 43
	return x[3]
}

// But the type of the map keys or values can be parameterized.
func Index4[T any](x map[int]T) T {
	var zero T
	x[2] = zero
	return x[3]
}

func test[T comparable](got, want T) {
	if got != want {
		panic(fmt.Sprintf("got %v, want %v", got, want))
	}
}

func main() {
	x := make([]int64, 4)
	x[3] = 2
	y := [5]int64{1, 2, 3, 4, 5}
	z := "abcd"
	w := make([]byte, 4)
	w[3] = 5
	v := make(map[int]int64)
	v[3] = 18

	test(Index1(x), int64(2))
	test(Index1(y), int64(4))
	test(Index2(z), byte(100))
	test(Index2(w), byte(5))
	test(Index2a(w), byte(5))
	test(Index3(v), int64(18))
	test(Index4(v), int64(18))
}
