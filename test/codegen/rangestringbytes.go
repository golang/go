// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// Issue 27148: a read-only range can reuse the byte slice's backing storage.
func RangeStringBytesCount(p []byte) int {
	n := 0
	// amd64:-`.*runtime.slicebytetostring`
	// arm64:-`.*runtime.slicebytetostring`
	for range string(p) {
		n++
	}
	return n
}

func RangeStringBytesSum(p []byte) int {
	n := 0
	// amd64:-`.*runtime.slicebytetostring`
	// arm64:-`.*runtime.slicebytetostring`
	for i, r := range string(p) {
		n += i + int(r)
	}
	return n
}

func RangeStringBytesControl(p []byte) int {
	n := 0
	// amd64:-`.*runtime.slicebytetostring`
	// arm64:-`.*runtime.slicebytetostring`
	for i, r := range string(p) {
		if r == 0 {
			break
		}
		if i&1 == 0 {
			continue
		}
		n += int(r)
	}
	return n
}

type rangeBytes []byte
type rangeText string

func RangeStringBytesNested(p rangeBytes) int {
	n, m := 0, 1
	// amd64:-`.*runtime.slicebytetostring`
	// arm64:-`.*runtime.slicebytetostring`
	for _, r := range rangeText(p) {
		for j := 0; j < 2; j++ {
			n, m = m, n+int(r)
		}
	}
	return n + m
}

// An assignment to an alias or an unknown call may change the source bytes.
func RangeStringBytesStore(p, alias []byte) int {
	n := 0
	// amd64:`.*runtime.slicebytetostring`
	// arm64:`.*runtime.slicebytetostring`
	for _, r := range string(p) {
		alias[0] = 0
		n += int(r)
	}
	return n
}

func RangeStringBytesCall(p []byte, f func()) int {
	n := 0
	// amd64:`.*runtime.slicebytetostring`
	// arm64:`.*runtime.slicebytetostring`
	for _, r := range string(p) {
		f()
		n += int(r)
	}
	return n
}

func RangeStringBytesClosure(p []byte) int {
	n := 0
	// amd64:`.*runtime.slicebytetostring`
	// arm64:`.*runtime.slicebytetostring`
	for _, r := range string(p) {
		func() { p[len(p)-1] = 0 }()
		n += int(r)
	}
	return n
}

func RangeStringBytesKey(p []byte, key *int) int {
	n := 0
	var r rune
	// amd64:`.*runtime.slicebytetostring`
	// arm64:`.*runtime.slicebytetostring`
	for *key, r = range string(p) {
		n += int(r)
	}
	return n
}

func RangeStringBytesValue(p []byte, value *rune) int {
	n := 0
	// amd64:`.*runtime.slicebytetostring`
	// arm64:`.*runtime.slicebytetostring`
	for _, *value = range string(p) {
		n++
	}
	return n
}

var rangeGlobal uint32

func RangeStringBytesGlobal(p []byte) int {
	n := 0
	// amd64:`.*runtime.slicebytetostring`
	// arm64:`.*runtime.slicebytetostring`
	for _, r := range string(p) {
		rangeGlobal = 0
		n += int(r)
	}
	return n
}

func RangeStringBytesAssignment(p []byte, f func() int) int {
	n := 0
	// amd64:`.*runtime.slicebytetostring`
	// arm64:`.*runtime.slicebytetostring`
	for _, r := range string(p) {
		n = n + int(r) + f()
	}
	return n
}

func RangeStringBytesInit(p []byte, f func() int) int {
	n := 0
	// amd64:`.*runtime.slicebytetostring`
	// arm64:`.*runtime.slicebytetostring`
	for _, r := range string(p) {
		if delta := f(); delta != 0 {
			n += delta
		}
		n += int(r)
	}
	return n
}
