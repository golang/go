// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//simdgen:category Reinterpretation

package spec

import "math"

// ToBits reinterprets the bits of each element of x as type {{.zE}}.
//
//specgen:name ToBits
//specgen:require xN=zN
func ToBits[xE Ints, xW Width, zE Uints](x Vec[xE, xW]) (z Vec[zE, xW]) {
	return map1[xE, xW, zE, xW](x, func(xe xE) zE { return zE(xe) })
}

// ToBitsFloat returns the IEEE 754 binary representation of each element of x.
//
//specgen:name ToBits
//specgen:require xN=zN
func ToBitsFloat[xE float32 | float64, xW Width, zE Uints](x Vec[xE, xW]) (z Vec[zE, xW]) {
	return map1[xE, xW, zE, xW](x, func(xe xE) zE {
		switch xe := any(xe).(type) {
		case float32:
			return zE(math.Float32bits(xe))
		case float64:
			return zE(math.Float64bits(xe))
		}
		panic("impossible type for xE")
	})
}

// ReshapeToUints reinterprets the bits of x as a {{.z}} vector. The least
// significant bit of element 0 is bit 0
//
//specgen:name ReshapeToUint{{.zN}}s
//specgen:require xN!=zN
func ReshapeToUints[xE Uints, xW Width, zE Uints](x Vec[xE, xW]) (z Vec[zE, xW]) {
	z = makeVec[zE, xW]()
	xN, zN := elemBits[xE](), elemBits[zE]()
	// Copy a byte at a time.
	for bit := 0; bit < width[xW](); bit += 8 {
		b := byte(x[bit/xN] >> (bit % xN))
		z[bit/zN] |= zE(b) << (bit % zN)
	}
	return z
}
