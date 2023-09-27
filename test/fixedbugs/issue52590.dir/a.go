// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import "unsafe"

func Append() {
	_ = append(appendArgs())
}

func Delete() {
	delete(deleteArgs())
}

func Print() {
	print(ints())
}

func Println() {
	println(ints())
}

func Complex() {
	_ = complex(float64s())
}

func Copy() {
	copy(slices())
}

func UnsafeAdd() {
	_ = unsafe.Add(unsafeAdd())
}

func UnsafeSlice() {
	_ = unsafe.Slice(unsafeSlice())
}

func appendArgs() ([]int, int) {
	return []int{}, 0
}

func deleteArgs() (map[int]int, int) {
	return map[int]int{}, 0
}

func ints() (int, int) {
	return 1, 1
}

func float64s() (float64, float64) {
	return 0, 0
}

func slices() ([]int, []int) {
	return []int{}, []int{}
}

func unsafeAdd() (unsafe.Pointer, int) {
	return nil, 0
}

func unsafeSlice() (*byte, int) {
	var p [10]byte
	return &p[0], 0
}
