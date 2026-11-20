// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd

// InternalCLMULImpl reports which archsimd carryless multiply the
// CarrylessMul methods of Uint64s use at the vector size in use: the
// archsimd type whose CarrylessMul methods they use, such as "Uint64x2", or
// "software".
func InternalCLMULImpl() string {
	return clmulImpl
}
