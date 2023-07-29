// compile

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 57955: ARM assembler fails to handle certain cases.

package main

func main() {
	Decode[int16](nil)
	Decode[uint16](nil)
	Decode[float64](nil)
}

func DecodeInt16(b []byte) (int16, int) {
	return 0, 0
}

func DecodeUint16(b []byte) (uint16, int) {
	return 0, 0
}

func DecodeFloat64(b []byte) (float64, int) {
	return 0, 0
}

func Decode[T any](b []byte) (T, int) {
	switch any(*new(T)).(type) {
	case int16:
		v, n := DecodeInt16(b)
		return any(v).(T), n
	case uint16:
		v, n := DecodeUint16(b)
		return any(v).(T), n
	case float64:
		v, n := DecodeFloat64(b)
		return any(v).(T), n
	default:
		panic("")
	}
}
