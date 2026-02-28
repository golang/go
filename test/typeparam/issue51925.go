// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

type IntLike interface {
	~int | ~int64 | ~int32 | ~int16 | ~int8
}

func Reduce[T any, U any, Uslice ~[]U](function func(T, U) T, sequence Uslice, initial T) T {
	result := initial
	for _, x := range sequence {
		result = function(result, x)
	}
	return result
}

func min[T IntLike](x, y T) T {
	if x < y {
		return x
	}
	return y
}

// Min returns the minimum element of `nums`.
func Min[T IntLike, NumSlice ~[]T](nums NumSlice) T {
	if len(nums) == 0 {
		return T(0)
	}
	return Reduce(min[T], nums, nums[0])
}

// VarMin is the variadic version of Min.
func VarMin[T IntLike](nums ...T) T {
	return Min(nums)
}

type myInt int

func main() {
	fmt.Println(VarMin(myInt(1), myInt(2)))

	seq := []myInt{1, 2}
	fmt.Println(Min(seq))
	fmt.Println(VarMin(seq...))
}
