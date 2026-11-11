// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//specgen:category Numeric conversion

package spec

import "math"

// Conversions that change the element size keep the total vector width the
// same, so the result has a different number of elements than x. A narrowing
// conversion stores its results in the low elements of the result and zeroes
// the rest. A widening conversion converts only the low elements of x and
// ignores the rest. For fixed-width vectors, this composes with GetLo to
// produce a narrower vector, and with HiToLo to convert the high elements.
// Because the receiver and result types imply the lane counts, the names of
// these conversions don't include one (ExtendToInt32, not ExtendLo4ToInt32),
// and the same name works for fixed-width and scalable vectors.
//
// TODO: The narrowing and widening doc paragraphs are duplicated across these
// functions. It would be nice to share them.

// ConvertFloatToFloat converts each element of x to {{.zE}}.
//
//	z[i] = {{.zE}}(x[i])
//
// {{if lt .zN .xN}}
// The results are rounded to the nearest representable value, or ±inf if they
// cannot be represented.
//
// The results are stored in the low {{lanes .x "x"}} elements of z.
// The remaining elements of z are zero.
// {{else}}
// Only the low {{lanes .z "z"}} elements of x are converted.
// The remaining elements of x are ignored.
// {{end}}
//
//specgen:name ConvertTo{{.zE | title}}
//specgen:require zE!=xE
func ConvertFloatToFloat[xE Floats, W Width, zE Floats](x Vec[xE, W]) (z Vec[zE, W]) {
	return map1[xE, W, zE, W](x, func(x xE) zE { return zE(x) })
}

// ConvertFloatToInt converts each element of x to {{.zE}}.
//
//	z[i] = {{.zE}}(x[i])
//
// When a conversion is inexact, this truncates the result (rounds toward zero).
// If the converted result would be outside the representable range, the result
// is architecture-dependent.
//
// {{if lt .zN .xN}}
// The results are stored in the low {{lanes .x "x"}} elements of z.
// The remaining elements of z are zero.
// {{else if gt .zN .xN}}
// Only the low {{lanes .z "z"}} elements of x are converted.
// The remaining elements of x are ignored.
// {{end}}
//
//specgen:name ConvertTo{{.zE | title}}
func ConvertFloatToInt[xE Floats, W Width, zE Ints | Uints](x Vec[xE, W]) (z Vec[Variation[zE], W]) {
	return map1[xE, W, Variation[zE], W](x, convertFloatToInt[xE, zE])
}

// convertFloatToInt converts x to zE, truncating toward zero. The result
// varies by platform if x is NaN or outside the range of zE.
func convertFloatToInt[xE Floats, zE Ints | Uints](x xE) Variation[zE] {
	// Compute representable range as [minLimit, maxLimit).
	minLimit := float64(minVal[zE]()) // 0 or -2^(N-1), always exact in float
	// The max value is a little trickier. Converting maxVal[zE]() directly to
	// xE or float64 would round 2^k - 1 up to 2^k when zE has more value bits
	// than the float significand (e.g. float32 -> int32 or float64 -> int64).
	// Compute 2^k directly.
	maxExp := elemBits[zE]()
	if isSigned[zE]() {
		maxExp--
	}
	maxLimit := math.Ldexp(1, maxExp) // maxVal[zE]() + 1 (2^N or 2^(N-1)), always exact in float

	// On AMD64, invalid conversions (NaN, ±Inf, or out of range) produce the
	// "integer indefinite" value: the minimum value for signed integers, or the
	// maximum value for unsigned integers. ARM64 and WASM instead map NaN to 0
	// and saturate out-of-range values (including ±Inf). Sadly, Intel did not
	// add saturating conversions until AVX10.2.
	amd64Indef := maxVal[zE]()
	if isSigned[zE]() {
		amd64Indef = minVal[zE]()
	}

	if math.IsNaN(float64(x)) {
		return perPlatform(map[Platform]zE{
			PlatformAMD64: amd64Indef,
			PlatformARM64: 0,
			PlatformWASM:  0,
		})
	}
	// Truncate toward zero before checking bounds, since fractional values
	// like 127.9 (for int8) or -0.5 (for uint8) truncate into range.
	// math.Trunc also preserves ±Inf.
	tx := math.Trunc(float64(x))
	// Check that x is in the representable range. Since tx is an integer
	// (or ±Inf), tx <= maxVal[zE]() is equivalent to tx < maxVal[zE]() + 1.
	if tx < minLimit {
		return perPlatform(map[Platform]zE{
			PlatformAMD64: amd64Indef, // Matches minVal if zE is signed.
			PlatformARM64: minVal[zE](),
			PlatformWASM:  minVal[zE](),
		})
	}
	if tx >= maxLimit {
		return perPlatform(map[Platform]zE{
			PlatformAMD64: amd64Indef, // Matches maxVal if zE is unsigned.
			PlatformARM64: maxVal[zE](),
			PlatformWASM:  maxVal[zE](),
		})
	}
	return exact(zE(tx))
}

// ConvertIntToFloat converts each element of x to {{.zE}}.
//
//	z[i] = {{.zE}}(x[i])
//
// {{if lt .zN .xN}}
// The results are stored in the low {{lanes .x "x"}} elements of z.
// The remaining elements of z are zero.
// {{else if gt .zN .xN}}
// Only the low {{lanes .z "z"}} elements of x are converted.
// The remaining elements of x are ignored.
// {{end}}
//
//specgen:name ConvertTo{{.zE | title}}
func ConvertIntToFloat[xE Ints | Uints, W Width, zE Floats](x Vec[xE, W]) (z Vec[zE, W]) {
	return map1[xE, W, zE, W](x, func(x xE) zE { return zE(x) })
}

// ConvertIntToUint converts each element of x to {{.zE}}.
//
//	z[i] = {{.zE}}(x[i])
//
// This wraps unrepresentable values.
//
//specgen:name ConvertTo{{.zE | title}}
//specgen:require zB!=xB zN=xN
func ConvertIntToUint[xE Ints | Uints, W Width, zE Ints | Uints](x Vec[xE, W]) (z Vec[zE, W]) {
	// This implements int->uint and uint->int. Within-kind conversions use
	// different names: see ExtendToZ, TruncToZ, and SaturateToZ.
	//
	// We limit this to keeping the same bit-width (which is always a no-op
	// conversion). Changing both the kind and the size raises questions about
	// sign-extension versus zero-extension that are better resolved by doing a
	// two-step conversion.
	return map1[xE, W, zE, W](x, func(x xE) zE { return zE(x) })
}

// ExtendToZ {{if eq .xB "int"}}sign{{else}}zero{{end}}-extends each element of x to {{.zE}}.
//
//	z[i] = {{.zE}}(x[i])
//
// Only the low {{lanes .z "z"}} elements of x are converted.
// The remaining elements of x are ignored.
//
//specgen:name ExtendTo{{.zE | title}}
//specgen:require zB=xB zN>xN
func ExtendToZ[xE Ints | Uints, W Width, zE Ints | Uints](x Vec[xE, W]) (z Vec[zE, W]) {
	return map1[xE, W, zE, W](x, func(x xE) zE { return zE(x) })
}

// TruncToZ converts each element of x to {{.zE}} by discarding its high
// bits.
//
//	z[i] = {{.zE}}(x[i])
//
// The results are stored in the low {{lanes .x "x"}} elements of z.
// The remaining elements of z are zero.
//
//specgen:name TruncTo{{.zE | title}}
//specgen:require zB=xB zN<xN
func TruncToZ[xE Ints | Uints, W Width, zE Ints | Uints](x Vec[xE, W]) (z Vec[zE, W]) {
	return map1[xE, W, zE, W](x, func(x xE) zE { return zE(x) })
}

// SaturateToZ converts each element of x to {{.zE}}, saturating values
// that are outside the range of {{.zE}}.
//
//	z[i] = sat(x[i])
//
// The results are stored in the low {{lanes .x "x"}} elements of z.
// The remaining elements of z are zero.
//
//specgen:name SaturateTo{{.zE | title}}
//specgen:require zB=xB zN<xN
func SaturateToZ[xE Ints | Uints, W Width, zE Ints | Uints](x Vec[xE, W]) (z Vec[zE, W]) {
	return map1[xE, W, zE, W](x, saturate[xE, zE])
}

// SaturateIntToUint converts each element of x to {{.zE}}, saturating values
// that are outside the range of {{.zE}}. Negative values become 0.
//
//	z[i] = sat(x[i])
//
// The results are stored in the low {{lanes .x "x"}} elements of z.
// The remaining elements of z are zero.
//
//specgen:name SaturateTo{{.zE | title}}
//specgen:require zN<xN
func SaturateIntToUint[xE Ints, W Width, zE Uints](x Vec[xE, W]) (z Vec[zE, W]) {
	// The name and meaning are the same as SaturateToZ: saturate to the
	// range of the result type. This is split out only because the
	// constraint language can't express "zB=xB or (xB=int and zB=uint)".
	return map1[xE, W, zE, W](x, saturate[xE, zE])
}

// ExtendLo* and ConvertLo* on 128-bit vectors are superseded by ExtendTo* and
// ConvertTo*, which have the same meaning there, work for scalable vectors, and
// don't put a lane count in the name. The APIs remain as deprecated wrappers,
// and ExtendLoLToZ and ConvertLoLToZ describe exactly those. Delete them when
// the wrappers are deleted.
//
// ExtendLoLToZWiden describes the ExtendLo* conversions from a 128-bit vector
// to a wider vector, which amd64 provides because each is a single instruction.
// These are also deprecated: they have no same-width equivalent, but can be
// written by placing x in the low part of a wider vector with SetLo and using
// ExtendTo*.

// ExtendLoLToZ {{if eq .xB "int"}}sign{{else}}zero{{end}}-extends the low {{.zL}} elements of x to {{.zE}}.
//
//	z[i] = {{.zE}}(x[i])
//
//specgen:name ExtendLo{{.zL}}To{{.zE | title}}
//specgen:require zB=xB zN>xN
//specgen:experiment
func ExtendLoLToZ[E Ints | Uints, zE Ints | Uints](x Vec[E, Width128]) (z Vec[zE, Width128]) {
	return map1[E, Width128, zE, Width128](x, func(x E) zE { return zE(x) })
}

// ExtendLoLToZWiden {{if eq .xB "int"}}sign{{else}}zero{{end}}-extends the low {{.zL}} elements of x to {{.zE}}.
//
//	z[i] = {{.zE}}(x[i])
//
//specgen:name ExtendLo{{.zL}}To{{.zE | title}}
//specgen:require zB=xB zL<xL zW>xW
//specgen:experiment
func ExtendLoLToZWiden[E Ints | Uints, zE Ints | Uints, zW FixedWidth](x Vec[E, Width128]) (z Vec[zE, zW]) {
	return map1[E, Width128, zE, zW](x, func(x E) zE { return zE(x) })
}

// ConvertLoLToZ converts the low-indexed {{.zL}} elements of x to {{.zE}}.
//
//	z[i] = {{.zE}}(x[i])
//
//specgen:name ConvertLo{{.zL}}To{{.zE | title}}
//specgen:require zN=xN*2
//specgen:experiment
func ConvertLoLToZ[E Floats, zE Floats](x Vec[E, Width128]) (z Vec[zE, Width128]) {
	return map1[E, Width128, zE, Width128](x, func(x E) zE { return zE(x) })
}
