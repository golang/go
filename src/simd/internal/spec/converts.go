// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spec

// ConvertToZ converts element values to {zE}. The result has the same number of lanes.
//
//specgen:name ConvertTo{zE}
//specgen:require zL=xL zE!=xE
func ConvertToZ[xE Nums, xW Width, zE Nums, zW Width](x Vec[xE, xW]) (z Vec[zE, zW]) {
	// Architectures are generally significantly more constrained in what they
	// will convert between, but this operation describes the full universe of
	// possible conversions.
	return map1[xE, xW, zE, zW](x, func(x xE) zE { return zE(x) })
}

// TODO: ExtendLo* and ConvertLo* don't work for scalable conversions because we
// put a literal zL in the name. We could instead just call them XLoToZ and drop
// the number since it's fully implied by the receiver type and the target type.
// Note that it's not necessarily the low *half*; for example,
// Uint8x16.ExtendLo2ToUint64 is the low eighth, but that's implied by going
// from uint8 to uint64 without changing the width.

// ExtendLoLToZ extends the lowest {zL} vector elements to {zE}.
//
//specgen:name ExtendLo{zL}To{zE}
//specgen:require zB=xB zN>xN
func ExtendLoLToZ[E Ints | Uints, W FixedWidth, zE Ints | Uints](x Vec[E, W]) (z Vec[zE, W]) {
	z = makeVec[zE, W]()
	for i := range z {
		z[i] = zE(x[i])
	}
	return z
}

// ConvertLoLToZ converts the low-indexed {zL} elements of x to {zE}.
//
//specgen:name ConvertLo{zL}To{zE}
//specgen:require zL<xL
func ConvertLoLToZ[E Nums, W FixedWidth, zE Floats](x Vec[E, W]) (z Vec[zE, W]) {
	panic("not implemented")
}
