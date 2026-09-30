// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//simdgen:category Math

package spec

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
