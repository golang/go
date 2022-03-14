// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type F { // ERROR expected type
	float64
} // ERROR expected declaration

func _[T F | int](x T) {
	_ = x == 0 // don't crash when recording type of 0
}

// test case from issue

type FloatType { // ERROR expected type
	float32 | float64
} // ERROR expected declaration

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
