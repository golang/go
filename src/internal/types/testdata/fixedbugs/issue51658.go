// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test checks syntax errors which differ between
// go/parser and the syntax package.
// TODO: consolidate eventually

package p

type F { // ERRORx "expected type|type declaration"
	float64
} // ERRORx "expected declaration|non-declaration statement"

func _[T F | int](x T) {
	_ = x == 0 // don't crash when recording type of 0
}

// test case from issue

type FloatType { // ERRORx "expected type|type declaration"
	float32 | float64
} // ERRORx "expected declaration|non-declaration statement"

type IntegerType interface {
	int8 | int16 | int32 | int64 | int |
		uint8 | uint16 | uint32 | uint64 | uint
}

type ComplexType interface {
	complex64 | complex128
}

type Number interface {
	FloatType | IntegerType | ComplexType
}

func GetDefaultNumber[T Number](value, defaultValue T) T {
	if value == 0 {
		return defaultValue
	}
	return value
}
