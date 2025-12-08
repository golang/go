// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package archsimd

// ClearAVXUpperBits clears the high bits of Y0-Y15 and Z0-Z15 registers.
// It is intended for transitioning from AVX to SSE, eliminating the
// performance penalties caused by false dependencies.
//
// Note: in the future the compiler may automatically generate the
// instruction, making this function unnecessary.
//
// Asm: VZEROUPPER, CPU Feature: AVX
func ClearAVXUpperBits()

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y
//
// Asm: VPTEST, CPU Feature: AVX
func (x Int8x16) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y
//
// Asm: VPTEST, CPU Feature: AVX
func (x Int8x32) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y
//
// Asm: VPTEST, CPU Feature: AVX
func (x Int16x8) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y
//
// Asm: VPTEST, CPU Feature: AVX
func (x Int16x16) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y
//
// Asm: VPTEST, CPU Feature: AVX
func (x Int32x4) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y
//
// Asm: VPTEST, CPU Feature: AVX
func (x Int32x8) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y
//
// Asm: VPTEST, CPU Feature: AVX
func (x Int64x2) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y
//
// Asm: VPTEST, CPU Feature: AVX
func (x Int64x4) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y
//
// Asm: VPTEST, CPU Feature: AVX
func (x Uint8x16) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y
//
// Asm: VPTEST, CPU Feature: AVX
func (x Uint8x32) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y
//
// Asm: VPTEST, CPU Feature: AVX
func (x Uint16x8) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y
//
// Asm: VPTEST, CPU Feature: AVX
func (x Uint16x16) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y
//
// Asm: VPTEST, CPU Feature: AVX
func (x Uint32x4) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y
//
// Asm: VPTEST, CPU Feature: AVX
func (x Uint32x8) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y
//
// Asm: VPTEST, CPU Feature: AVX
func (x Uint64x2) IsZero() bool

// IsZero returns true if all elements of x are zeros.
//
// This method compiles to VPTEST x, x.
// x.And(y).IsZero() and x.AndNot(y).IsZero() will be optimized to VPTEST x, y
//
// Asm: VPTEST, CPU Feature: AVX
func (x Uint64x4) IsZero() bool
