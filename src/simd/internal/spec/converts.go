// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//specgen:category Numeric conversion

package spec

import "math"

// TODO: The notes on ConvertXToZ apply to anything that does width rounding.
// Should we have a quick template for that? Or even just add it automatically?

// ConvertFloatToFloat converts each element of x to {{.zE}}.
//
// {{ if lt .zN .xN }}
// The results are rounded to the nearest representable value, or ±inf if they
// cannot be represented.
// {{ end }}
//
// {{if lt .xL .zL}}
// The low {{.xL}} elements of the result are set. The rest are zero.
// {{end}}
//
//specgen:name ConvertTo{{.zE | title}}
//specgen:require z={zB}{zN}x{xL} zE!=xE
func ConvertFloatToFloat[xE Floats, xW Width, zE Floats, zW Width](x Vec[xE, xW]) (z Vec[zE, zW]) {
	return map1[xE, xW, zE, zW](x, func(x xE) zE { return zE(x) })
}

// ConvertFloatToInt converts each element of x to {{.zE}}.
//
// When a conversion is inexact, this truncates the result (rounds toward zero).
// If the converted result would be outside the representable range, the result
// is architecture-dependent.
//
// {{if lt .xL .zL}}
// The low {{.xL}} elements of the result are set. The rest are zero.
// {{end}}
//
//specgen:name ConvertTo{{.zE | title}}
//specgen:require z={zB}{zN}x{xL}
func ConvertFloatToInt[xE Floats, xW Width, zE Ints | Uints, zW Width](x Vec[xE, xW]) (z Vec[Variation[zE], zW]) {
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

	return map1[xE, xW, Variation[zE], zW](x, func(x xE) Variation[zE] {
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
				PlatformAMD64: amd64Indef, // Matches maxVal is zE is unsigned.
				PlatformARM64: maxVal[zE](),
				PlatformWASM:  maxVal[zE](),
			})
		}
		return exact(zE(tx))
	})
}

// ConvertIntToFloat converts each element of x to {{.zE}}.
//
// {{if lt .xL .zL}}
// The low {{.xL}} elements of the result are set. The rest are zero.
// {{end}}
//
//specgen:name ConvertTo{{.zE | title}}
//specgen:require z={zB}{zN}x{xL}
func ConvertIntToFloat[xE Ints | Uints, xW Width, zE Floats, zW Width](x Vec[xE, xW]) (z Vec[zE, zW]) {
	return map1[xE, xW, zE, zW](x, func(x xE) zE { return zE(x) })
}

// ConvertIntToUint converts each element of x to {{.zE}}.
//
// This wraps unrepresentable values.
//
// {{if lt .xL .zL}}
// The low {{.xL}} elements of the result are set. The rest are zero.
// {{end}}
//
//specgen:name ConvertTo{{.zE | title}}
//specgen:require z={zB}{zN}x{xL} zB!=xB zN=xN
func ConvertIntToUint[xE Ints | Uints, xW Width, zE Ints | Uints, zW Width](x Vec[xE, xW]) (z Vec[zE, zW]) {
	// This implements int->uint and uint->int. Within-kind conversions use
	// different names.
	//
	// Currently we also limit this to keeping the same bit-width (which is
	// always a no-op conversion). We could lift that restriction, but that
	// raises complex questions about sign-extension versus zero-extension that
	// are better resolved by doing a two-step conversion. Saturating versions
	// may make more sense.
	return map1[xE, xW, zE, zW](x, func(x xE) zE { return zE(x) })
}

// TODO: ExtendLo* and ConvertLo* don't work for scalable conversions because we
// put a literal zL in the name. We could instead just call them XLoToZ and drop
// the number since it's fully implied by the receiver type and the target type.
// Note that it's not necessarily the low *half*; for example,
// Uint8x16.ExtendLo2ToUint64 is the low eighth, but that's implied by going
// from uint8 to uint64 without changing the width.

// ExtendLoLToZ {{if eq .xB "int"}}sign-{{else}}zero-{{end}}extends
// the lowest {{.zL}} vector elements to {{.zE}}.
//
//specgen:name ExtendLo{{.zL}}To{{.zE | title}}
//specgen:require zB=xB zN>xN
func ExtendLoLToZ[E Ints | Uints, W FixedWidth, zE Ints | Uints](x Vec[E, W]) (z Vec[zE, W]) {
	z = makeVec[zE, W]()
	for i := range z {
		z[i] = zE(x[i])
	}
	return z
}

// ExtendHiLToZ {{if eq .xB "int"}}sign-{{else}}zero-{{end}}extends the high-indexed {{.zL}} elements of x to {{.zE}}.
//
//	z[i] = {{.zE}}(x[{{.zL}}+i])
//
//specgen:name ExtendHi{{.zL}}To{{.zE | title}}
//specgen:require zB=xB zN=xN*2
func ExtendHiLToZ[E Ints | Uints, zE Ints | Uints](x Vec[E, Width128]) (z Vec[zE, Width128]) {
	z = makeVec[zE, Width128]()
	half := z.len()
	for i := range z {
		z[i] = zE(x[half+i])
	}
	return z
}

// ConvertLoLToZ converts the low-indexed {{.zL}} elements of x to {{.zE}}.
//
//	z[i] = {{.zE}}(x[i])
//
//specgen:name ConvertLo{{.zL}}To{{.zE | title}}
//specgen:require zL<xL
func ConvertLoLToZ[E Nums, W FixedWidth, zE Floats](x Vec[E, W]) (z Vec[zE, W]) {
	z = makeVec[zE, W]()
	for i := range z {
		z[i] = zE(x[i])
	}
	return z
}
