// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//simdgen:category Math

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
//	z = {x[0]-x[1], y[0]-y[1]}
//
// {{else if eq .xL 4}}
//
//	z = {sat(x[0]-x[1]), sat(x[2]-x[3]), sat(y[0]-y[1]), sat(y[2]-y[3])}
//
// {{else}}
//
//	z = {x[0]-x[1], x[2]-x[3], ..., y[0]-y[1], y[2]-y[3], ...}
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

// Div divides x by y elementwise.
//
//	z[i] = x[i] / y[i]
//
// {{if eq .xB "float"}}
//
// Division by zero does not panic, and the result follows IEEE 754. That is,
// dividing a non-zero value by zero results in +/- infinity, and dividing zero
// by zero results in NaN.
//
// {{else}}
//
// The result is rounded toward zero (truncated division), just like Go's /
// operator.{{if eq .xB "int"}} Also like Go's / operator, dividing Min{{title .xE}}
// by -1 results in Min{{title .xE}}, since the true result is unrepresentable.{{end}}
// Unlike Go's / operator, division by zero does not panic and instead produces
// a zero result for that element.
//
// {{end}}
func Div[E Elt, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
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
//specgen:commutative
//specgen:require z={xB}{xN*2}x{xL/2}
func DotProductPairs[E Nums, W Width, zE Nums](x, y Vec[E, W]) (z Vec[zE, W]) {
	// TODO: How do we handle/specify overflow? x86 only supports this on signed
	// types, and the only case that can overflow is if all four elements are
	// MinInt16 (in which case the true result is MaxInt32+1, which wraps around
	// to MinInt32). Unsigned types can overflow much more readily.
	//
	// Maybe we just leave overflow unspecified (or "architecture dependent").
	// In which case, we probably need a way to communicate that in the spec
	// (designated panic?).
	//
	// We might also need a way to constraint this to same-signed E and zE,
	// which the constraint language doesn't currently have a way to say, but we
	// could add as a built-in projection function in the syntax.
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

// Max returns the elementwise maximum of x and y.
//
//	z[i] = max(x[i], y[i])
func Max[E Elt, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	// Note that this is NOT marked as commutative because its handling of NaN
	// on AMD64 is non-commutative.

	// TODO: Platforms do not agree on how to handle NaNs.
	//
	// AMD64 AVX (VMAXPS/VMINPS): If either input is NaN or if inputs are +0.0
	// and −0.0, the instruction selects the second operand (making it
	// non-commutative).
	//
	// ARM64 NEON (VFMAX/VFMIN): Follows IEEE 754-2008 maxNum/minNum (returns
	// the numerical operand if one is NaN).
	//
	// WASM (F32x4Max/F32x4Min): Follows IEEE 754-2019 maximum/minimum.
	//
	// simd_emulated.go: Uses Go's built-in max/min (max(x, NaN) = NaN, and
	// preserves +0.0 vs −0.0).
	return map2[E, W, E, W](x, y, func(x, y E) E {
		return max(x, y)
	})
}

// Min returns the elementwise minimum of x and y.
//
//	z[i] = min(x[i], y[i])
func Min[E Elt, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	// TODO: Same issue as Max.
	return map2[E, W, E, W](x, y, func(x, y E) E {
		return min(x, y)
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

// MulAdd returns x * y + z elementwise.
//
//	w[i] = x[i] * y[i] + z[i]
func MulAdd[E Elt, W Width](x, y, z Vec[E, W]) (w Vec[E, W]) {
	w = makeVec[E, W]()
	for i := range x {
		w[i] = x[i]*y[i] + z[i]
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

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
func ReduceSum[E Elt, W Width](x Vec[E, W]) (z E) {
	// TODO: Determine and document the precision of the result.
	for _, xi := range x {
		z += xi
	}
	return z
}

// Sqrt returns the elementwise square root of x.
//
//	z[i] = sqrt(x[i])
func Sqrt[E Floats, W Width](x Vec[E, W]) (z Vec[E, W]) {
	// TODO: Determine and document the precision of the result.
	//
	// TODO: Document behavior for special float values.
	return map1[E, W, E, W](x, func(x E) E {
		return E(math.Sqrt(float64(x)))
	})
}
