// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// These tests check that mapaccess calls are not used.
// Issues #23661 and #24364.

func mapCompoundAssignmentInt8() {
	m := make(map[int8]int8, 0)
	var k int8 = 0

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] += 67

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] -= 123

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] *= 45

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] |= 78

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] ^= 89

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] <<= 9

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] >>= 10

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k]++

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k]--
}

func mapCompoundAssignmentInt32() {
	m := make(map[int32]int32, 0)
	var k int32 = 0

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] += 67890

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] -= 123

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] *= 456

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] |= 78

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] ^= 89

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] <<= 9

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] >>= 10

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k]++

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k]--
}

func mapCompoundAssignmentInt64() {
	m := make(map[int64]int64, 0)
	var k int64 = 0

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] += 67890

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] -= 123

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] *= 456

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] |= 78

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] ^= 89

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] <<= 9

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] >>= 10

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k]++

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k]--
}

func mapCompoundAssignmentComplex128() {
	m := make(map[complex128]complex128, 0)
	var k complex128 = 0

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] += 67890

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] -= 123

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] *= 456

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k]++

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k]--
}

func mapCompoundAssignmentString() {
	m := make(map[string]string, 0)
	var k string = "key"

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] += "value"
}

var sinkAppend bool

func mapAppendAssignmentInt8() {
	m := make(map[int8][]int8, 0)
	var k int8 = 0

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] = append(m[k], 1)

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] = append(m[k], 1, 2, 3)

	a := []int8{7, 8, 9, 0}

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] = append(m[k], a...)

	// Exceptions

	// 386:".*mapaccess"
	// amd64:".*mapaccess"
	// arm:".*mapaccess"
	// arm64:".*mapaccess"
	m[k] = append(a, m[k]...)

	// 386:".*mapaccess"
	// amd64:".*mapaccess"
	// arm:".*mapaccess"
	// arm64:".*mapaccess"
	sinkAppend, m[k] = !sinkAppend, append(m[k], 99)

	// 386:".*mapaccess"
	// amd64:".*mapaccess"
	// arm:".*mapaccess"
	// arm64:".*mapaccess"
	m[k] = append(m[k+1], 100)
}

func mapAppendAssignmentInt32() {
	m := make(map[int32][]int32, 0)
	var k int32 = 0

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] = append(m[k], 1)

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] = append(m[k], 1, 2, 3)

	a := []int32{7, 8, 9, 0}

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] = append(m[k], a...)

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k+1] = append(m[k+1], a...)

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[-k] = append(m[-k], a...)

	// Exceptions

	// 386:".*mapaccess"
	// amd64:".*mapaccess"
	// arm:".*mapaccess"
	// arm64:".*mapaccess"
	m[k] = append(a, m[k]...)

	// 386:".*mapaccess"
	// amd64:".*mapaccess"
	// arm:".*mapaccess"
	// arm64:".*mapaccess"
	sinkAppend, m[k] = !sinkAppend, append(m[k], 99)

	// 386:".*mapaccess"
	// amd64:".*mapaccess"
	// arm:".*mapaccess"
	// arm64:".*mapaccess"
	m[k] = append(m[k+1], 100)
}

func mapAppendAssignmentInt64() {
	m := make(map[int64][]int64, 0)
	var k int64 = 0

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] = append(m[k], 1)

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] = append(m[k], 1, 2, 3)

	a := []int64{7, 8, 9, 0}

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] = append(m[k], a...)

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k+1] = append(m[k+1], a...)

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[-k] = append(m[-k], a...)

	// Exceptions

	// 386:".*mapaccess"
	// amd64:".*mapaccess"
	// arm:".*mapaccess"
	// arm64:".*mapaccess"
	m[k] = append(a, m[k]...)

	// 386:".*mapaccess"
	// amd64:".*mapaccess"
	// arm:".*mapaccess"
	// arm64:".*mapaccess"
	sinkAppend, m[k] = !sinkAppend, append(m[k], 99)

	// 386:".*mapaccess"
	// amd64:".*mapaccess"
	// arm:".*mapaccess"
	// arm64:".*mapaccess"
	m[k] = append(m[k+1], 100)
}

func mapAppendAssignmentComplex128() {
	m := make(map[complex128][]complex128, 0)
	var k complex128 = 0

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] = append(m[k], 1)

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] = append(m[k], 1, 2, 3)

	a := []complex128{7, 8, 9, 0}

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] = append(m[k], a...)

	// Exceptions

	// 386:".*mapaccess"
	// amd64:".*mapaccess"
	// arm:".*mapaccess"
	// arm64:".*mapaccess"
	m[k] = append(a, m[k]...)

	// 386:".*mapaccess"
	// amd64:".*mapaccess"
	// arm:".*mapaccess"
	// arm64:".*mapaccess"
	sinkAppend, m[k] = !sinkAppend, append(m[k], 99)

	// 386:".*mapaccess"
	// amd64:".*mapaccess"
	// arm:".*mapaccess"
	// arm64:".*mapaccess"
	m[k] = append(m[k+1], 100)
}

func mapAppendAssignmentString() {
	m := make(map[string][]string, 0)
	var k string = "key"

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] = append(m[k], "1")

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] = append(m[k], "1", "2", "3")

	a := []string{"7", "8", "9", "0"}

	// 386:-".*mapaccess"
	// amd64:-".*mapaccess"
	// arm:-".*mapaccess"
	// arm64:-".*mapaccess"
	m[k] = append(m[k], a...)

	// Exceptions

	// 386:".*mapaccess"
	// amd64:".*mapaccess"
	// arm:".*mapaccess"
	// arm64:".*mapaccess"
	m[k] = append(a, m[k]...)

	// 386:".*mapaccess"
	// amd64:".*mapaccess"
	// arm:".*mapaccess"
	// arm64:".*mapaccess"
	sinkAppend, m[k] = !sinkAppend, append(m[k], "99")

	// 386:".*mapaccess"
	// amd64:".*mapaccess"
	// arm:".*mapaccess"
	// arm64:".*mapaccess"
	m[k] = append(m[k+"1"], "100")
}
