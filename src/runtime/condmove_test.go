// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"math/rand"
	"testing"
)

var (
	testData32      = make([]uint32, 1000)
	testData16      = make([]uint16, 1000)
	testDataInt     = make([]int, 1000)
	testDataUintptr = make([]uintptr, 1000)
	testDataFloat64 = make([]float64, 1000)
	testDataBool    = make([]bool, 1000)
)

func init() {
	for i := range testData32 {
		if i%4 == 0 {
			testData32[i] = 0
		} else if i%4 == 1 {
			testData32[i] = 1
		} else if i%4 == 2 {
			testData32[i] = uint32(rand.Intn(1000))
		} else {
			testData32[i] = 0xFFFFFFFF
		}
		testData16[i] = uint16(testData32[i])
		testDataInt[i] = int(testData32[i])
		testDataUintptr[i] = uintptr(testData32[i])
		testDataFloat64[i] = float64(rand.Float64() * 1000.0)
		testDataBool[i] = (rand.Intn(2) == 0)
	}
}

func BenchmarkCmovXOR32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testData32); j++ {
			_ = cmovXOR32(testData32[j], testData32[(j+1)%len(testData32)])
		}
	}
}

func BenchmarkCmovPseudoRandom32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testData32); j++ {
			_ = cmovPseudoRandom32(testData32[j], testData32[(j+1)%len(testData32)])
		}
	}
}

func BenchmarkCmovAddZero(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testDataInt); j++ {
			_ = cmoveAddZero(testDataInt[j], testDataInt[(j+1)%len(testDataInt)], testDataInt[(j+2)%len(testDataInt)])
		}
	}
}

func BenchmarkCmovAddNonZero(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testDataInt); j++ {
			_ = cmoveAddNonZero(testDataInt[j], testDataInt[(j+1)%len(testDataInt)], testDataInt[(j+2)%len(testDataInt)])
		}
	}
}

func BenchmarkCmovSubZero(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testDataInt); j++ {
			_ = cmoveSubZero(testDataInt[j], testDataInt[(j+1)%len(testDataInt)], testDataInt[(j+2)%len(testDataInt)])
		}
	}
}

func BenchmarkCmovOrZero(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testDataInt); j++ {
			_ = cmoveOrZero(testDataInt[j], testDataInt[(j+1)%len(testDataInt)], testDataInt[(j+2)%len(testDataInt)])
		}
	}
}

func BenchmarkCmovXorZero(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testDataInt); j++ {
			_ = cmoveXorZero(testDataInt[j], testDataInt[(j+1)%len(testDataInt)], testDataInt[(j+2)%len(testDataInt)])
		}
	}
}

func BenchmarkCmoveAddConstZero(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testDataInt); j++ {
			_ = cmoveAddConstZero(testDataInt[j], testDataInt[(j+1)%len(testDataInt)])
		}
	}
}

func BenchmarkCmoveAddConstNonZero(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testDataInt); j++ {
			_ = cmoveAddConstNonZero(testDataInt[j], testDataInt[(j+1)%len(testDataInt)])
		}
	}
}

func BenchmarkCmovInc(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testDataBool); j++ {
			_ = cmovInc(testDataBool[j], testDataInt[j], testDataInt[(j+1)%len(testDataInt)])
		}
	}
}

func BenchmarkCmovInv(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testDataBool); j++ {
			_ = cmovInv(testDataBool[j], testDataInt[j], testDataInt[(j+1)%len(testDataInt)])
		}
	}
}

func BenchmarkCmovNeg(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testDataBool); j++ {
			_ = cmovNeg(testDataBool[j], testDataInt[j], testDataInt[(j+1)%len(testDataInt)])
		}
	}
}

func BenchmarkCmovSetm(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testDataBool); j++ {
			_ = cmovSetm(testDataBool[j])
		}
	}
}

func BenchmarkCmovZero1(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testDataBool); j++ {
			_ = cmovZero1(testDataBool[j])
		}
	}
}

//go:noinline
func cmovXOR32(x, y uint32) uint32 {
	result := x
	if ((x ^ y) & 1) != 0 {
		result = y
	}
	return result
}

//go:noinline
func cmovPseudoRandom32(x, y uint32) uint32 {
	result := x
	hash := x ^ y
	hash ^= hash >> 16
	hash ^= hash >> 8
	hash ^= hash >> 4
	hash ^= hash >> 2
	hash ^= hash >> 1
	if (hash & 1) != 0 {
		result = y
	}
	return result
}

//go:noinline
func cmoveAddZero(cond, a, b int) int {
	if cond == 0 {
		a += b
	}
	return a
}

//go:noinline
func cmoveAddNonZero(cond, a, b int) int {
	if cond != 0 {
		a += b
	}
	return a
}

//go:noinline
func cmoveSubZero(cond, a, b int) int {
	if cond == 0 {
		a -= b
	}
	return a
}

//go:noinline
func cmoveOrZero(cond, a, b int) int {
	if cond == 0 {
		a |= b
	}
	return a
}

//go:noinline
func cmoveXorZero(cond, a, b int) int {
	if cond == 0 {
		a ^= b
	}
	return a
}

//go:noinline
func cmoveAddConstZero(cond, a int) int {
	if cond == 0 {
		a += 42
	}
	return a
}

//go:noinline
func cmoveAddConstNonZero(cond, a int) int {
	if cond != 0 {
		a += 42
	}
	return a
}

//go:noinline
func cmovInc(cond bool, a, b int) int {
	var x0 int
	if cond {
		x0 = a
	} else {
		x0 = b + 1
	}
	return x0
}

//go:noinline
func cmovInv(cond bool, a, b int) int {
	var x0 int
	if cond {
		x0 = a
	} else {
		x0 = ^b
	}
	return x0
}

//go:noinline
func cmovNeg(cond bool, a, b int) int {
	var x0 int
	if cond {
		x0 = a
	} else {
		x0 = -b
	}
	return x0
}

//go:noinline
func cmovSetm(cond bool) int {
	var x0 int
	if cond {
		x0 = -1
	} else {
		x0 = 0
	}
	return x0
}

//go:noinline
func cmovZero1(cond bool) int {
	var x int
	if cond {
		x = 182
	}
	return x
}
