// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//specgen:category Math

package spec

import "math"

// AbsFloat returns the elementwise absolute value of x.
//
//specgen:name Abs
func AbsFloat[E Floats, W Width](x Vec[E, W]) (z Vec[E, W]) {
	return map1[E, W, E, W](x, func(x E) E {
		// Use math.Abs to handle float corner cases.
		switch x := any(x).(type) {
		case float64:
			return E(math.Abs(x))
		case float32:
			return E(math.Abs(float64(x)))
		}
		panic("unexpected type")
	})
}

// AbsInt returns the elementwise absolute value of x.
//
//specgen:name Abs
//specgen:require z=Uint{xN}x{xL}
func AbsInt[E Ints, W Width, zE Uints](x Vec[E, W]) (z Vec[zE, W]) {
	// We return an unsigned result because it can represent the whole range of
	// the absolute value (the signed type can't represent abs(minVal[E])).
	// Conveniently, this aligns with hardware operations as long as they either
	// produce an unsigned result or "wrap". In the latter case, wrapping in the
	// signed domain followed by reinterpreting the result as unsigned is the
	// same as just doing an unsigned absolute value.
	return map1[E, W, zE, W](x, func(x E) zE {
		if x < 0 {
			// If x is MinIntN, -x == x (this can be viewed as wrapping in the
			// signed domain), and uintN(-x) == uintN(x) == abs(x).
			return zE(-x)
		}
		return zE(x)
	})
}

// Add adds x and y elementwise.
//
//	z[i] = x[i] + y[i]
//
//specgen:commutative
func Add[E Nums, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	return map2[E, W, E, W](x, y, func(x, y E) E { return x + y })
}

// Sub subtracts y from x elementwise.
//
//	z[i] = x[i] - y[i]
func Sub[E Nums, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	return map2[E, W, E, W](x, y, func(x, y E) E { return x - y })
}

// AddSaturated adds x and y elementwise with saturation.
//
//	z[i] = sat(x[i] + y[i])
//
//specgen:commutative
func AddSaturated[E Ints | Uints, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	return map2[E, W, E, W](x, y, func(x, y E) E { return addSaturated(x, y) })
}

// SubSaturated subtracts y from x elementwise with saturation.
//
//	z[i] = sat(x[i] - y[i])
func SubSaturated[E Ints | Uints, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	return map2[E, W, E, W](x, y, func(x, y E) E { return subSaturated(x, y) })
}

// ConcatAddPairs horizontally adds adjacent pairs of elements in x and y and
// returns the concatenated result.
//
// {{if eq .xL 2}}
//
//	z = {x[0]+x[1], y[0]+y[1]}
//
// {{else if eq .xL 4}}
//
//	z = {x[0]+x[1], x[2]+x[3], y[0]+y[1], y[2]+y[3]}
//
// {{else}}
//
//	z = {x[0]+x[1], x[2]+x[3], ..., y[0]+y[1], y[2]+y[3], ...}
//
// {{end}}
func ConcatAddPairs[E Nums, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	z = makeVec[E, W]()
	half := z.len() / 2
	for i := 0; i < half; i++ {
		z[i] = x[2*i] + x[2*i+1]
		z[half+i] = y[2*i] + y[2*i+1]
	}
	return z
}

// ConcatSubPairs horizontally subtracts adjacent pairs of elements in x and y
// and returns the concatenated result.
//
// {{if eq .xL 2}}
//
//	z = {x[0]-x[1], y[0]-y[1]}
//
// {{else if eq .xL 4}}
//
//	z = {x[0]-x[1], x[2]-x[3], y[0]-y[1], y[2]-y[3]}
//
// {{else}}
//
//	z = {x[0]-x[1], x[2]-x[3], ..., y[0]-y[1], y[2]-y[3], ...}
//
// {{end}}
func ConcatSubPairs[E Nums, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	z = makeVec[E, W]()
	half := z.len() / 2
	for i := 0; i < half; i++ {
		z[i] = x[2*i] - x[2*i+1]
		z[half+i] = y[2*i] - y[2*i+1]
	}
	return z
}

// ConcatAddPairsSaturated horizontally adds adjacent pairs of elements in x and
// y with saturation and returns the concatenated result.
//
// {{if eq .xL 2}}
//
//	z = {sat(x[0]+x[1]), sat(y[0]+y[1])}
//
// {{else if eq .xL 4}}
//
//	z = {sat(x[0]+x[1]), sat(x[2]+x[3]), sat(y[0]+y[1]), sat(y[2]+y[3])}
//
// {{else}}
//
//	z = {sat(x[0]+x[1]), sat(x[2]+x[3]), ..., sat(y[0]+y[1]), sat(y[2]+y[3]), ...}
//
// {{end}}
func ConcatAddPairsSaturated[E Ints | Uints, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	z = makeVec[E, W]()
	half := z.len() / 2
	for i := 0; i < half; i++ {
		z[i] = addSaturated(x[2*i], x[2*i+1])
		z[half+i] = addSaturated(y[2*i], y[2*i+1])
	}
	return z
}

// ConcatSubPairsSaturated horizontally subtracts adjacent pairs of elements in
// x and y with saturation and returns the concatenated result.
//
// {{if eq .xL 2}}
//
//	z = {sat(x[0]-x[1]), sat(y[0]-y[1])}
//
// {{else if eq .xL 4}}
//
//	z = {sat(x[0]-x[1]), sat(x[2]-x[3]), sat(y[0]-y[1]), sat(y[2]-y[3])}
//
// {{else}}
//
//	z = {sat(x[0]-x[1]), sat(x[2]-x[3]), ..., sat(y[0]-y[1]), sat(y[2]-y[3]), ...}
//
// {{end}}
func ConcatSubPairsSaturated[E Ints | Uints, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	z = makeVec[E, W]()
	half := z.len() / 2
	for i := 0; i < half; i++ {
		z[i] = subSaturated(x[2*i], x[2*i+1])
		z[half+i] = subSaturated(y[2*i], y[2*i+1])
	}
	return z
}

// ConcatAddPairsGrouped divides x, y, and z into groups of 128 bits and
// performs [ConcatAddPairs] on each group.
func ConcatAddPairsGrouped[E Nums, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	return grouped128(ConcatAddPairs, x, y)
}

// ConcatSubPairsGrouped divides x, y, and z into groups of 128 bits and
// performs [ConcatSubPairs] on each group.
func ConcatSubPairsGrouped[E Nums, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	return grouped128(ConcatSubPairs, x, y)
}

// ConcatAddPairsSaturatedGrouped divides x, y, and z into groups of 128 bits
// and performs [ConcatAddPairsSaturated] on each group.
func ConcatAddPairsSaturatedGrouped[E Ints | Uints, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	return grouped128(ConcatAddPairsSaturated, x, y)
}

// ConcatSubPairsSaturatedGrouped divides x, y, and z into groups of 128 bits
// and performs [ConcatSubPairsSaturated] on each group.
func ConcatSubPairsSaturatedGrouped[E Ints | Uints, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	return grouped128(ConcatSubPairsSaturated, x, y)
}

// Average returns the elementwise average of x and y, rounded toward +∞.
//
//	z[i] = (x[i] + y[i] + 1) / 2
func Average[E Ints | Uints, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	return map2[E, W, E, W](x, y, func(x, y E) E {
		// This formula works on both signed and unsigned, and never overflows.
		return (x >> 1) + (y >> 1) + ((x | y) & 1)
	})
}

// DivInts divides x by y elementwise.
//
//	z[i] = x[i] / y[i]
//
// The result is rounded toward zero (truncated division), just like Go's /
// operator.{{if eq .xB "int"}} Also like Go's / operator, dividing Min{{title .xE}}
// by -1 results in Min{{title .xE}}, since the true result is unrepresentable.{{end}}
// Unlike Go's / operator, division by zero does not panic and instead produces
// a zero result for that element.
//
//specgen:name Div
func DivInts[E Ints | Uints, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	return map2[E, W, E, W](x, y, func(x, y E) E {
		if y == 0 {
			return 0
		}
		return x / y
	})
}

// DivFloats divides x by y elementwise.
//
//	z[i] = x[i] / y[i]
//
// Division by zero does not panic, and the result follows IEEE 754. That is,
// dividing a non-zero value by zero results in +/- infinity, and dividing zero
// by zero results in NaN.
//
//specgen:name Div
func DivFloats[E Floats, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	return map2[E, W, E, W](x, y, func(x, y E) E {
		return x / y
	})
}

// DotProductPairs multiplies corresponding elements of x and y, and sums
// adjacent pairs, returning a vector of half as many elements, each with twice
// the input element size.
//
//	w[i] = x[i] * y[i]        // Double width
//	z[j] = w[2*j] + w[2*j+1]
//
// While the products can never overflow, the sum wraps on overflow.
// {{if eq .xB "int"}}
// This happens in exactly one situation: if all four input elements are Min{{title .xE}}.
// {{end}}
//
//specgen:commutative
//specgen:require z={xB}{xN*2}x{xL/2}
func DotProductPairs[E Ints | Uints, W Width, zE Nums](x, y Vec[E, W]) (z Vec[zE, W]) {
	z = makeVec[zE, W]()
	for i := range z {
		z[i] = zE(x[2*i])*zE(y[2*i]) + zE(x[2*i+1])*zE(y[2*i+1])
	}
	return z
}

// DotProductPairsSaturated multiplies corresponding elements of x and y, and
// sums adjacent pairs, all with saturation. It returns a vector of half as many
// elements, each with twice the input element size.
//
//	w[i] = sat(x[i] * y[i])        // Double width
//	z[j] = sat(w[2*j] + w[2*j+1])
//
//specgen:commutative
//specgen:require y=Int{xN}x{xL} z=Int{xN*2}x{xL/2}
func DotProductPairsSaturated[xE Uints, xW Width, yE Ints, zE Ints](x Vec[xE, xW], y Vec[yE, xW]) (z Vec[zE, xW]) {
	z = makeVec[zE, xW]()
	for i := range z {
		a := mulSaturatedUSS64(uint64(x[2*i]), int64(y[2*i]))
		b := mulSaturatedUSS64(uint64(x[2*i+1]), int64(y[2*i+1]))
		z[i] = saturateS[zE](addSaturatedSSS64(a, b))
	}
	return z
}

// MaxInts returns the elementwise maximum of x and y.
//
//	z[i] = max(x[i], y[i])
//
//specgen:name Max
//specgen:commutative
func MaxInts[E Ints | Uints, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	return map2[E, W, E, W](x, y, func(x, y E) E {
		return max(x, y)
	})
}

// MaxFloats returns the elementwise maximum of x and y.
//
//	z[i] = max(x[i], y[i])
//
// The handling of NaN and signed zeroes is architecture-dependent. ARM64 and
// WASM follow the IEEE 754-2019 maximum algorithm, which also agrees with Go's
// max built-in. AMD64 handles these differently. Notably, AMD64 is
// non-commutative.
//
//	x     y     IEEE  AMD64
//	----- ----- ----- -----
//	NaN   y     NaN   y
//	x     NaN   NaN   NaN
//	-Inf  y     y     y
//	x     -Inf  x     x
//	+Inf  y     +Inf  +Inf
//	x     +Inf  +Inf  +Inf
//	+0    -0    +0    -0
//	-0    +0    +0    +0
//
//specgen:name Max
func MaxFloats[E Floats, W Width](x, y Vec[E, W]) (z Vec[Variation[E], W]) {
	// Note that this is NOT marked as commutative because its handling of NaN
	// and signed zeroes on AMD64 is non-commutative.
	//
	// AMD64 AVX (VMAXPS/VMINPS): Computes `if x > y { x } else { y }`. If
	// either input is NaN or if inputs are +0.0 and −0.0, the instruction
	// selects the second operand y (making it non-commutative).
	//
	// ARM64 NEON/SVE (VFMAX/VFMIN, ZFMAX/ZFMIN) and WASM (F32x4Max/F32x4Min):
	// Follow IEEE 754-2019 maximum/minimum (like Go's built-in max/min,
	// propagating NaN and ordering −0.0 < +0.0).
	return map2[E, W, Variation[E], W](x, y, func(x, y E) Variation[E] {
		amd64 := y
		if x > y {
			amd64 = x
		}
		ieee := max(x, y)
		if equalExact(amd64, ieee) {
			return exact(ieee)
		}
		return perPlatform(map[Platform]E{
			PlatformAMD64: amd64,
			PlatformARM64: ieee,
			PlatformWASM:  ieee,
		})
	})
}

// MinInts returns the elementwise minimum of x and y.
//
//	z[i] = min(x[i], y[i])
//
//specgen:name Min
//specgen:commutative
func MinInts[E Ints | Uints, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	return map2[E, W, E, W](x, y, func(x, y E) E {
		return min(x, y)
	})
}

// MinFloats returns the elementwise minimum of x and y.
//
//	z[i] = min(x[i], y[i])
//
// The handling of NaN and signed zeroes is architecture-dependent. ARM64 and
// WASM follow the IEEE 754-2019 minimum algorithm, which also agrees with Go's
// min built-in. AMD64 handles these differently. Notably, AMD64 is
// non-commutative.
//
//	x     y     IEEE  AMD64
//	----- ----- ----- -----
//	NaN   y     NaN   y
//	x     NaN   NaN   NaN
//	-Inf  y     -Inf  -Inf
//	x     -Inf  -Inf  -Inf
//	+Inf  y     y     y
//	x     +Inf  x     x
//	+0    -0    -0    -0
//	-0    +0    -0    +0
//
//specgen:name Min
func MinFloats[E Floats, W Width](x, y Vec[E, W]) (z Vec[Variation[E], W]) {
	// See MaxFloats for platform details.
	return map2[E, W, Variation[E], W](x, y, func(x, y E) Variation[E] {
		amd64 := y
		if x < y {
			amd64 = x
		}
		ieee := min(x, y)
		if equalExact(amd64, ieee) {
			return exact(ieee)
		}
		return perPlatform(map[Platform]E{
			PlatformAMD64: amd64,
			PlatformARM64: ieee,
			PlatformWASM:  ieee,
		})
	})
}

// Mul returns the elementwise product of x and y.
//
//	z[i] = x[i] * y[i]
//
//specgen:commutative
func Mul[E Elt, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	return map2[E, W, E, W](x, y, func(x, y E) E {
		return x * y
	})
}

// MulAddInts returns x * y + z elementwise.
//
//	w[i] = x[i] * y[i] + z[i]
//
//specgen:name MulAdd
func MulAddInts[E Ints | Uints, W Width](x, y, z Vec[E, W]) (w Vec[E, W]) {
	w = makeVec[E, W]()
	for i := range x {
		w[i] = x[i]*y[i] + z[i]
	}
	return w
}

// MulAddFloats returns x * y + z elementwise.
//
//	w[i] = x[i] * y[i] + z[i]
//
// This performs a fused multiply-add, computing x[i]*y[i] + z[i] with a single
// rounding rather than two and no intermediate overflow, except on wasm, where
// it rounds x[i]*y[i] before adding z[i].
//
//specgen:name MulAdd
func MulAddFloats[E Floats, W Width](x, y, z Vec[E, W]) (w Vec[Variation[E], W]) {
	// Every platform that can reasonably implement SIMD has fused vector
	// multiply-add instructions, except WASM: SIMD128 has none, so WASM
	// implements MulAdd as a multiply followed by an add. (Relaxed SIMD's
	// relaxed_madd may or may not fuse, depending on the host, and not all
	// runtimes support it.)
	//
	// Go has the option of compiling x*y + z using FMA, but only does so if FMA
	// support is guaranteed by the hardware version (e.g., GOAMD64>=3). To
	// guarantee the right result, we have to do it ourself.
	w = makeVec[Variation[E], W]()
	for i := range x {
		var fused E
		switch xi := any(x[i]).(type) {
		case float32:
			fused = E(math.FMA32(xi, float32(y[i]), float32(z[i])))
		case float64:
			fused = E(math.FMA(xi, float64(y[i]), float64(z[i])))
		}
		// The explicit conversion prevents the Go compiler from fusing the
		// multiply and add.
		unfused := E(x[i]*y[i]) + z[i]
		if equalExact(fused, unfused) {
			w[i] = exact(fused)
			continue
		}
		w[i] = perPlatformDefault(fused, map[Platform]E{
			PlatformWASM: unfused,
		})
	}
	return w
}

// Neg returns the elementwise negation of x.
//
//	z[i] = -x[i]
func Neg[E Ints | Floats, W Width](x Vec[E, W]) (z Vec[E, W]) {
	return map1[E, W, E, W](x, func(x E) E {
		return -x
	})
}

// ReduceSumInts returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
//
//specgen:name ReduceSum
func ReduceSumInts[E Ints | Uints, W Width](x Vec[E, W]) (z E) {
	// Integer addition wraps and is associative, so the order doesn't matter.
	for _, xi := range x {
		z += xi
	}
	return z
}

// ReduceSumFloats returns the scalar sum of the elements of x.
//
// {{if eq .xL 2}}
//
//	z = x[0] + x[1]
//
// {{else}}
//
// This uses repeated pairwise summation, which may affect the final precision
// of the result.
//
// {{if eq .xL 4}}
//
//	z = (x[0] + x[1]) + (x[2] + x[3])
//
// {{else}}
//
//	z = ((x[0] + x[1]) + (x[2] + x[3])) + ...
//
// {{end}}
//
// {{end}}
//
//specgen:name ReduceSum
func ReduceSumFloats[E Floats, W Width](x Vec[E, W]) (z E) {
	// Vector lengths are powers of two, so every level pairs up evenly.
	tmp := make([]E, x.len()/2, maxScalableWidth/elemBits[E]()/2)
	for i := range tmp {
		tmp[i] = x[2*i] + x[2*i+1]
	}
	for len(tmp) > 1 {
		half := len(tmp) / 2
		for i := range half {
			tmp[i] = tmp[2*i] + tmp[2*i+1]
		}
		tmp = tmp[:half]
	}
	return tmp[0]
}

// ReduceMax returns the largest element of x.
//
//	z = max(x[0], x[1], ...)
func ReduceMax[E Elt, W Width](x Vec[E, W]) (z E) {
	// TODO: Do we need per-platform handling of non-finites? Since AMD64 is the
	// only platform that's weird about non-finite values and it doesn't have
	// ReduceMax, maybe we don't have to, but if we ever want to emulate this on
	// AMD64, we'd have to decide whether to match AMD64's Max or Go/IEEE Max.
	z = x[0]
	for _, xi := range x[1:] {
		z = max(z, xi)
	}
	return z
}

// Sqrt returns the elementwise square root of x.
//
//	z[i] = sqrt(x[i])
//
// The result is correctly rounded. Sqrt(+Inf) = +Inf, Sqrt(±0) = ±0, and
// Sqrt(x) is NaN if x < 0 or x is NaN. The sign and payload of a NaN result
// are unspecified.
func Sqrt[E Floats, W Width](x Vec[E, W]) (z Vec[E, W]) {
	// For float32, computing the square root in float64 and rounding to
	// float32 is still correctly rounded: double rounding is harmless for
	// square root when the wider format has at least 2p+2 bits of precision,
	// and 53 >= 2*24+2.
	return map1[E, W, E, W](x, func(x E) E {
		return E(math.Sqrt(float64(x)))
	})
}
