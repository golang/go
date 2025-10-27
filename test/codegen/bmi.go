// asmcheck

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func andn64(x, y int64) int64 {
	// amd64/v3:"ANDNQ"
	return x &^ y
}

func andn32(x, y int32) int32 {
	// amd64/v3:"ANDNL"
	return x &^ y
}

func blsi64(x int64) int64 {
	// amd64/v3:"BLSIQ"
	return x & -x
}

func blsi32(x int32) int32 {
	// amd64/v3:"BLSIL"
	return x & -x
}

func blsmsk64(x int64) int64 {
	// amd64/v3:"BLSMSKQ"
	return x ^ (x - 1)
}

func blsmsk32(x int32) int32 {
	// amd64/v3:"BLSMSKL"
	return x ^ (x - 1)
}

func blsr64(x int64) int64 {
	// amd64/v3:"BLSRQ"
	return x & (x - 1)
}

func blsr32(x int32) int32 {
	// amd64/v3:"BLSRL"
	return x & (x - 1)
}

func isPowerOfTwo64(x int64) bool {
	// amd64/v3:"BLSRQ" -"TESTQ" -"CALL"
	return blsr64(x) == 0
}

func isPowerOfTwo32(x int32) bool {
	// amd64/v3:"BLSRL" -"TESTL" -"CALL"
	return blsr32(x) == 0
}

func isPowerOfTwoSelect64(x, a, b int64) int64 {
	var r int64
	// amd64/v3:"BLSRQ" -"TESTQ" -"CALL"
	if isPowerOfTwo64(x) {
		r = a
	} else {
		r = b
	}
	// amd64/v3:"CMOVQEQ" -"TESTQ" -"CALL"
	return r * 2 // force return blocks joining
}

func isPowerOfTwoSelect32(x, a, b int32) int32 {
	var r int32
	// amd64/v3:"BLSRL" -"TESTL" -"CALL"
	if isPowerOfTwo32(x) {
		r = a
	} else {
		r = b
	}
	// amd64/v3:"CMOVLEQ" -"TESTL" -"CALL"
	return r * 2 // force return blocks joining
}

func isPowerOfTwoBranch64(x int64, a func(bool), b func(string)) {
	// amd64/v3:"BLSRQ" -"TESTQ" -"CALL"
	if isPowerOfTwo64(x) {
		a(true)
	} else {
		b("false")
	}
}

func isPowerOfTwoBranch32(x int32, a func(bool), b func(string)) {
	// amd64/v3:"BLSRL" -"TESTL" -"CALL"
	if isPowerOfTwo32(x) {
		a(true)
	} else {
		b("false")
	}
}

func isNotPowerOfTwo64(x int64) bool {
	// amd64/v3:"BLSRQ" -"TESTQ" -"CALL"
	return blsr64(x) != 0
}

func isNotPowerOfTwo32(x int32) bool {
	// amd64/v3:"BLSRL" -"TESTL" -"CALL"
	return blsr32(x) != 0
}

func isNotPowerOfTwoSelect64(x, a, b int64) int64 {
	var r int64
	// amd64/v3:"BLSRQ" -"TESTQ" -"CALL"
	if isNotPowerOfTwo64(x) {
		r = a
	} else {
		r = b
	}
	// amd64/v3:"CMOVQNE" -"TESTQ" -"CALL"
	return r * 2 // force return blocks joining
}

func isNotPowerOfTwoSelect32(x, a, b int32) int32 {
	var r int32
	// amd64/v3:"BLSRL" -"TESTL" -"CALL"
	if isNotPowerOfTwo32(x) {
		r = a
	} else {
		r = b
	}
	// amd64/v3:"CMOVLNE" -"TESTL" -"CALL"
	return r * 2 // force return blocks joining
}

func isNotPowerOfTwoBranch64(x int64, a func(bool), b func(string)) {
	// amd64/v3:"BLSRQ" -"TESTQ" -"CALL"
	if isNotPowerOfTwo64(x) {
		a(true)
	} else {
		b("false")
	}
}

func isNotPowerOfTwoBranch32(x int32, a func(bool), b func(string)) {
	// amd64/v3:"BLSRL" -"TESTL" -"CALL"
	if isNotPowerOfTwo32(x) {
		a(true)
	} else {
		b("false")
	}
}

func sarx64(x, y int64) int64 {
	// amd64/v3:"SARXQ"
	return x >> y
}

func sarx32(x, y int32) int32 {
	// amd64/v3:"SARXL"
	return x >> y
}

func sarx64_load(x []int64, i int) int64 {
	// amd64/v3: `SARXQ [A-Z]+[0-9]*, \([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), [A-Z]+[0-9]*`
	s := x[i] >> (i & 63)
	// amd64/v3: `SARXQ [A-Z]+[0-9]*, 8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), [A-Z]+[0-9]*`
	s = x[i+1] >> (s & 63)
	return s
}

func sarx32_load(x []int32, i int) int32 {
	// amd64/v3: `SARXL [A-Z]+[0-9]*, \([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), [A-Z]+[0-9]*`
	s := x[i] >> (i & 63)
	// amd64/v3: `SARXL [A-Z]+[0-9]*, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), [A-Z]+[0-9]*`
	s = x[i+1] >> (s & 63)
	return s
}

func shlrx64(x, y uint64) uint64 {
	// amd64/v3:"SHRXQ"
	s := x >> y
	// amd64/v3:"SHLXQ"
	s = s << y
	return s
}

func shlrx32(x, y uint32) uint32 {
	// amd64/v3:"SHRXL"
	s := x >> y
	// amd64/v3:"SHLXL"
	s = s << y
	return s
}

func shlrx64_load(x []uint64, i int, s uint64) uint64 {
	// amd64/v3: `SHRXQ [A-Z]+[0-9]*, \([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), [A-Z]+[0-9]*`
	s = x[i] >> i
	// amd64/v3: `SHLXQ [A-Z]+[0-9]*, 8\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*8\), [A-Z]+[0-9]*`
	s = x[i+1] << s
	return s
}

func shlrx32_load(x []uint32, i int, s uint32) uint32 {
	// amd64/v3: `SHRXL [A-Z]+[0-9]*, \([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), [A-Z]+[0-9]*`
	s = x[i] >> i
	// amd64/v3: `SHLXL [A-Z]+[0-9]*, 4\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*4\), [A-Z]+[0-9]*`
	s = x[i+1] << s
	return s
}
