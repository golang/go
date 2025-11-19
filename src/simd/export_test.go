// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

// This exposes some internal interfaces to simd_test.

package simd

func (x Int64x2) ExportTestConcatSelectedConstant(indices uint8, y Int64x2) Int64x2 {
	return x.concatSelectedConstant(indices, y)
}

func (x Float64x4) ExportTestConcatSelectedConstantGrouped(indices uint8, y Float64x4) Float64x4 {
	return x.concatSelectedConstantGrouped(indices, y)
}

func (x Float32x4) ExportTestConcatSelectedConstant(indices uint8, y Float32x4) Float32x4 {
	return x.concatSelectedConstant(indices, y)
}

func (x Int32x4) ExportTestConcatSelectedConstant(indices uint8, y Int32x4) Int32x4 {
	return x.concatSelectedConstant(indices, y)
}

func (x Uint32x8) ExportTestConcatSelectedConstantGrouped(indices uint8, y Uint32x8) Uint32x8 {
	return x.concatSelectedConstantGrouped(indices, y)
}

func (x Int32x8) ExportTestConcatSelectedConstantGrouped(indices uint8, y Int32x8) Int32x8 {
	return x.concatSelectedConstantGrouped(indices, y)
}

func (x Int32x8) ExportTestTern(table uint8, y Int32x8, z Int32x8) Int32x8 {
	return x.tern(table, y, z)
}

func (x Int32x4) ExportTestTern(table uint8, y Int32x4, z Int32x4) Int32x4 {
	return x.tern(table, y, z)
}

func ExportTestCscImm4(a, b, c, d uint8) uint8 {
	return cscimm4(a, b, c, d)
}

const (
	LLLL = _LLLL
	HLLL = _HLLL
	LHLL = _LHLL
	HHLL = _HHLL
	LLHL = _LLHL
	HLHL = _HLHL
	LHHL = _LHHL
	HHHL = _HHHL
	LLLH = _LLLH
	HLLH = _HLLH
	LHLH = _LHLH
	HHLH = _HHLH
	LLHH = _LLHH
	HLHH = _HLHH
	LHHH = _LHHH
	HHHH = _HHHH
)
