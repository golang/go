// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

func min(left int, right int) int {
	if left < right {
		return left
	}
	return right
}

func minInt32(left int32, right int32) int32 {
	if left < right {
		return left
	}
	return right
}

func max(left int, right int) int {
	if left > right {
		return left
	}
	return right
}

func fillInts(a []int, value int) {
	for i := range a {
		a[i] = value
	}
}

func fillInt32s(a []int32, value int32) {
	for i := range a {
		a[i] = value
	}
}

func fillBytes(a []byte, value byte) {
	for i := range a {
		a[i] = value
	}
}

func fillInt8s(a []int8, value int8) {
	for i := range a {
		a[i] = value
	}
}

func fillUint8s(a []uint8, value uint8) {
	for i := range a {
		a[i] = value
	}
}

func copyInt8s(dst []int8, src []int8) int {
	cnt := min(len(dst), len(src))
	for i := 0; i < cnt; i++ {
		dst[i] = src[i]
	}
	return cnt
}

func copyUint8s(dst []uint8, src []uint8) int {
	cnt := min(len(dst), len(src))
	for i := 0; i < cnt; i++ {
		dst[i] = src[i]
	}
	return cnt
}
