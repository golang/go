// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || wasm || arm64)

package bridge

type _simd struct {
	_ [0]func(*_simd) *_simd
}

// Int8s represents a 128-bit vector of 16 int8 elements.
type Int8s struct {
	_    _simd
	a, b uint64
}

// Int16s represents a 128-bit vector of 8 int16 elements.
type Int16s struct {
	_    _simd
	a, b uint64
}

// Int32s represents a 128-bit vector of 4 int32 elements.
type Int32s struct {
	_    _simd
	a, b uint64
}

// Int64s represents a 128-bit vector of 2 int64 elements.
type Int64s struct {
	_    _simd
	a, b uint64
}

// Uint8s represents a 128-bit vector of 16 uint8 elements.
type Uint8s struct {
	_    _simd
	a, b uint64
}

// Uint16s represents a 128-bit vector of 8 uint16 elements.
type Uint16s struct {
	_    _simd
	a, b uint64
}

// Uint32s represents a 128-bit vector of 4 uint32 elements.
type Uint32s struct {
	_    _simd
	a, b uint64
}

// Uint64s represents a 128-bit vector of 2 uint64 elements.
type Uint64s struct {
	_    _simd
	a, b uint64
}

// Float32s represents a 128-bit vector of 4 float32 elements.
type Float32s struct {
	_    _simd
	a, b uint64
}

// Float64s represents a 128-bit vector of 2 float64 elements.
type Float64s struct {
	_    _simd
	a, b uint64
}

// Mask8s represents a 128-bit mask vector for 16 int8/uint8 elements.
type Mask8s struct {
	_    _simd
	a, b uint64
}

// Mask16s represents a 128-bit mask vector for 8 int16/uint16 elements.
type Mask16s struct {
	_    _simd
	a, b uint64
}

// Mask32s represents a 128-bit mask vector for 4 int32/uint32/float32 elements.
type Mask32s struct {
	_    _simd
	a, b uint64
}

// Mask64s represents a 128-bit mask vector for 2 int64/uint64/float64 elements.
type Mask64s struct {
	_    _simd
	a, b uint64
}
