// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package archsimd

// Logos fault-suppressing slice loads (SPEC Sec 15.2, T-303; #22).
//
// The Load<T>Part loaders of slice_gen_amd64.go load a full vector from
// the address of the first element and then clear the lanes beyond
// len(s) with an AND. The load itself still touches the bytes past the
// end of the slice, so a slice that ends at a page boundary faults. The
// loaders below issue the masked load directly (VPMASKMOVD and
// VPMASKMOVQ for 32 and 64-bit lanes, the zeroing K-masked VMOVDQU8 and
// VMOVDQU16 for 8 and 16-bit lanes): a masked-off lane never touches
// memory and never faults. The intrinsic ANDs the caller mask with the
// lanes below len(s), so no mask enables a read past s[len(s)-1]. The
// clamp costs a length splat, an iota compare, and one AND per call.
//
// Each loader has no Go body. The compiler replaces every call with the
// generic LoadMasked32, LoadMasked64, LoadMasked16, or LoadMasked8 SSA
// operation on the first element address (ssagen/simd_logos.go), so the
// instruction appears in the caller and no call remains.

// LoadFloat32x8SliceMasked loads a Float32x8 from the slice s at the lanes
// enabled by mask and zeroes the others. A disabled lane reads no
// memory. Lanes at or past len(s) are disabled even when the mask
// enables them: they read no memory and produce zero.
//
// Asm: VPMASKMOVD, CPU Feature: AVX2
//
//go:noescape
func LoadFloat32x8SliceMasked(s []float32, mask Mask32x8) Float32x8

// LoadInt32x8SliceMasked loads a Int32x8 from the slice s at the lanes
// enabled by mask and zeroes the others. A disabled lane reads no
// memory. Lanes at or past len(s) are disabled even when the mask
// enables them: they read no memory and produce zero.
//
// Asm: VPMASKMOVD, CPU Feature: AVX2
//
//go:noescape
func LoadInt32x8SliceMasked(s []int32, mask Mask32x8) Int32x8

// LoadUint32x8SliceMasked loads a Uint32x8 from the slice s at the lanes
// enabled by mask and zeroes the others. A disabled lane reads no
// memory. Lanes at or past len(s) are disabled even when the mask
// enables them: they read no memory and produce zero.
//
// Asm: VPMASKMOVD, CPU Feature: AVX2
//
//go:noescape
func LoadUint32x8SliceMasked(s []uint32, mask Mask32x8) Uint32x8

// LoadFloat64x4SliceMasked loads a Float64x4 from the slice s at the lanes
// enabled by mask and zeroes the others. A disabled lane reads no
// memory. Lanes at or past len(s) are disabled even when the mask
// enables them: they read no memory and produce zero.
//
// Asm: VPMASKMOVQ, CPU Feature: AVX2
//
//go:noescape
func LoadFloat64x4SliceMasked(s []float64, mask Mask64x4) Float64x4

// LoadInt64x4SliceMasked loads a Int64x4 from the slice s at the lanes
// enabled by mask and zeroes the others. A disabled lane reads no
// memory. Lanes at or past len(s) are disabled even when the mask
// enables them: they read no memory and produce zero.
//
// Asm: VPMASKMOVQ, CPU Feature: AVX2
//
//go:noescape
func LoadInt64x4SliceMasked(s []int64, mask Mask64x4) Int64x4

// LoadUint64x4SliceMasked loads a Uint64x4 from the slice s at the lanes
// enabled by mask and zeroes the others. A disabled lane reads no
// memory. Lanes at or past len(s) are disabled even when the mask
// enables them: they read no memory and produce zero.
//
// Asm: VPMASKMOVQ, CPU Feature: AVX2
//
//go:noescape
func LoadUint64x4SliceMasked(s []uint64, mask Mask64x4) Uint64x4

// LoadInt16x32SliceMasked loads a Int16x32 from the slice s at the lanes
// enabled by mask and zeroes the others. A disabled lane reads no
// memory. Lanes at or past len(s) are disabled even when the mask
// enables them: they read no memory and produce zero.
//
// Asm: VMOVDQU16, CPU Feature: AVX512
//
//go:noescape
func LoadInt16x32SliceMasked(s []int16, mask Mask16x32) Int16x32

// LoadUint16x32SliceMasked loads a Uint16x32 from the slice s at the lanes
// enabled by mask and zeroes the others. A disabled lane reads no
// memory. Lanes at or past len(s) are disabled even when the mask
// enables them: they read no memory and produce zero.
//
// Asm: VMOVDQU16, CPU Feature: AVX512
//
//go:noescape
func LoadUint16x32SliceMasked(s []uint16, mask Mask16x32) Uint16x32

// LoadInt8x64SliceMasked loads a Int8x64 from the slice s at the lanes
// enabled by mask and zeroes the others. A disabled lane reads no
// memory. Lanes at or past len(s) are disabled even when the mask
// enables them: they read no memory and produce zero.
//
// Asm: VMOVDQU8, CPU Feature: AVX512
//
//go:noescape
func LoadInt8x64SliceMasked(s []int8, mask Mask8x64) Int8x64

// LoadUint8x64SliceMasked loads a Uint8x64 from the slice s at the lanes
// enabled by mask and zeroes the others. A disabled lane reads no
// memory. Lanes at or past len(s) are disabled even when the mask
// enables them: they read no memory and produce zero.
//
// Asm: VMOVDQU8, CPU Feature: AVX512
//
//go:noescape
func LoadUint8x64SliceMasked(s []uint8, mask Mask8x64) Uint8x64
