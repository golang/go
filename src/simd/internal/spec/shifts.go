// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//specgen:category Shifts

package spec

import "math/bits"

// ScaleSaturated multiplies each element of x by a power of two given by scale,
// with saturation.
//
// Positive exponents scale up (shift left with saturation). Negative exponents
// scale down ({{if eq .xB "int"}}arithmetic{{else}}logical{{end}} shift right).
//
// {{if eq .xN 8}}
//
//	z[i] = sat(x[i] * 2^scale[i])
//
// {{else}}
//
// scale has the same total width as x, but its elements are int8. Only every
// k'th element of scale is used, where k={{.xN}}/{{.scaleN}}. The rest are ignored.
//
//	z[i] = sat(x[i] * 2^scale[i*k])
//
// {{end}}
//
//specgen:experiment
func ScaleSaturated[E Ints | Uints, W Width](x Vec[E, W], scale Vec[int8, W]) (z Vec[E, W]) {
	k := elemBits[E]() / 8
	z = makeVec[E, W]()
	for i := range z {
		z[i] = scaleSaturated(x[i], scale[i*k])
	}
	return z
}

// {{if eq .xB "int" -}}
// ShiftAllRight arithmetically shifts each element of x right by shift bits.
// As in Go, if shift is greater than or equal to the element width, the
// result is 0 or -1, depending on the sign of the element.
// {{- else -}}
// ShiftAllRight logically shifts each element of x right by shift bits.
// As in Go, if shift is greater than or equal to the element width, the
// result is 0.
// {{- end}}
//
//	z[i] = x[i] >> shift
func ShiftAllRight[E Ints | Uints, W Width](x Vec[E, W], shift uint64) (z Vec[E, W]) {
	return map1[E, W, E, W](x, func(v E) E {
		return v >> shift
	})
}

// ShiftAllLeft shifts each element of x left by shift bits.
// As in Go, if shift is greater than or equal to the element width, the
// result is 0.
//
//	z[i] = x[i] << shift
func ShiftAllLeft[E Ints | Uints, W Width](x Vec[E, W], shift uint64) (z Vec[E, W]) {
	return map1[E, W, E, W](x, func(v E) E {
		return v << shift
	})
}

// {{if eq .xB "int" -}}
// ShiftRight arithmetically shifts x[i] right by shift[i] bits.
// As in Go, if shift[i] is greater than or equal to the element width, the
// result is 0 or -1, depending on the sign of the element.
// {{- else -}}
// ShiftRight logically shifts x[i] right by shift[i] bits.
// As in Go, if shift[i] is greater than or equal to the element width, the
// result is 0.
// {{- end}}
//
//	z[i] = x[i] >> shift[i]
//
//specgen:require shift=Uint{xN}x{xL}
func ShiftRight[E Ints | Uints, W Width, yE Uints](x Vec[E, W], shift Vec[yE, W]) (z Vec[E, W]) {
	z = makeVec[E, W]()
	for i := range z {
		z[i] = x[i] >> shift[i]
	}
	return z
}

// ShiftLeft shifts x[i] left by shift[i] bits.
// As in Go, if shift[i] is greater than or equal to the element width, the
// result is 0.
//
//	z[i] = x[i] << shift[i]
//
//specgen:require shift=Uint{xN}x{xL}
func ShiftLeft[E Ints | Uints, W Width, yE Uints](x Vec[E, W], shift Vec[yE, W]) (z Vec[E, W]) {
	z = makeVec[E, W]()
	for i := range z {
		z[i] = x[i] << shift[i]
	}
	return z
}

// RotateAllLeft rotates each element of x left by shift bits.
//
//	z[i] = rotateLeft(x[i], shift)
func RotateAllLeft[E Uints, W Width](x Vec[E, W], shift uint64) (z Vec[E, W]) {
	// Compute mod before casting to int.
	k := int(shift % uint64(elemBits[E]()))
	return map1[E, W, E, W](x, func(x E) E {
		switch any(x).(type) {
		case int8, uint8:
			return E(bits.RotateLeft8(uint8(x), k))
		case int16, uint16:
			return E(bits.RotateLeft16(uint16(x), k))
		case int32, uint32:
			return E(bits.RotateLeft32(uint32(x), k))
		case int64, uint64:
			return E(bits.RotateLeft64(uint64(x), k))
		}
		panic("unreachable")
	})
}

// RotateLeft rotates x left by shift bits, elementwise.
//
//	z[i] = rotateLeft(x[i], shift[i])
func RotateLeft[E Uints, W Width](x, shift Vec[E, W]) (z Vec[E, W]) {
	return map2[E, W, E, W](x, shift, func(x, shift E) E {
		// Compute mod before casting to int.
		k := int(shift % E(elemBits[E]()))
		switch any(x).(type) {
		case int8, uint8:
			return E(bits.RotateLeft8(uint8(x), k))
		case int16, uint16:
			return E(bits.RotateLeft16(uint16(x), k))
		case int32, uint32:
			return E(bits.RotateLeft32(uint32(x), k))
		case int64, uint64:
			return E(bits.RotateLeft64(uint64(x), k))
		}
		panic("unreachable")
	})
}

// RotateAllRight rotates each element of x right by shift bits.
//
//	z[i] = rotateRight(x[i], shift)
func RotateAllRight[E Uints, W Width](x Vec[E, W], shift uint64) (z Vec[E, W]) {
	// Compute mod before casting to int.
	k := int(shift % uint64(elemBits[E]()))
	return map1[E, W, E, W](x, func(x E) E {
		switch any(x).(type) {
		case int8, uint8:
			return E(bits.RotateLeft8(uint8(x), -k))
		case int16, uint16:
			return E(bits.RotateLeft16(uint16(x), -k))
		case int32, uint32:
			return E(bits.RotateLeft32(uint32(x), -k))
		case int64, uint64:
			return E(bits.RotateLeft64(uint64(x), -k))
		}
		panic("unreachable")
	})
}

// RotateRight rotates x right by shift bits, elementwise.
//
//	z[i] = rotateRight(x[i], shift[i])
func RotateRight[E Uints, W Width](x, shift Vec[E, W]) (z Vec[E, W]) {
	return map2[E, W, E, W](x, shift, func(x, shift E) E {
		// Compute mod before casting to int.
		k := int(shift % E(elemBits[E]()))
		switch any(x).(type) {
		case int8, uint8:
			return E(bits.RotateLeft8(uint8(x), -k))
		case int16, uint16:
			return E(bits.RotateLeft16(uint16(x), -k))
		case int32, uint32:
			return E(bits.RotateLeft32(uint32(x), -k))
		case int64, uint64:
			return E(bits.RotateLeft64(uint64(x), -k))
		}
		panic("unreachable")
	})
}

// ShiftAllLeftConcatModN shifts x[i] left by shift%{{.xN}}, filling the emptied
// low bits with the high bits of y[i]. That is, it shifts the double-width
// concatenation of x[i] (high) and y[i] (low) left and keeps the high {{.xN}}
// bits.
//
//	s = shift % {{.xN}}
//	z[i] = x[i]<<s | y[i]>>({{.xN}}-s)
//
//specgen:name ShiftAllLeftConcatMod{{.xN}}
//specgen:experiment
func ShiftAllLeftConcatModN[E Uints, W Width](x, y Vec[E, W], shift uint64) (z Vec[E, W]) {
	// TODO: Should this just be called ShiftAllLeftConcatMod, without the element width, like LookupMod?
	return map2[E, W, E, W](x, y, func(x, y E) E {
		return concatShiftLeft(x, y, shift)
	})
}

// ShiftAllRightConcatModN shifts x[i] right by shift%{{.xN}}, filling the
// emptied high bits with the low bits of y[i]. That is, it shifts the
// double-width concatenation of y[i] (high) and x[i] (low) right and keeps the
// low {{.xN}} bits.
//
//	s = shift % {{.xN}}
//	z[i] = x[i]>>s | y[i]<<({{.xN}}-s)
//
//specgen:name ShiftAllRightConcatMod{{.xN}}
//specgen:experiment
func ShiftAllRightConcatModN[E Uints, W Width](x, y Vec[E, W], shift uint64) (z Vec[E, W]) {
	// TODO: Maybe "Concat" is the wrong term? Everywhere else we use it for
	// concatenating vectors, not bits, and even at the bit level
	// ShiftAllRightConcatModN is concat(y, x), which is different from all of
	// the other concat operations.
	return map2[E, W, E, W](x, y, func(x, y E) E {
		return concatShiftRight(x, y, shift)
	})
}

// ShiftLeftConcatModN shifts x[i] left by shift[i]%{{.xN}}, filling the
// emptied low bits with the high bits of y[i]. That is, it shifts the
// double-width concatenation of x[i] (high) and y[i] (low) left and keeps the
// high {{.xN}} bits.
//
//	s[i] = shift[i] % {{.xN}}
//	z[i] = x[i]<<s[i] | y[i]>>({{.xN}}-s[i])
//
//specgen:name ShiftLeftConcatMod{{.xN}}
//specgen:require shift=Uint{xN}x{xL}
//specgen:experiment
func ShiftLeftConcatModN[E Uints, W Width, sE Uints](x, y Vec[E, W], shift Vec[sE, W]) (z Vec[E, W]) {
	z = makeVec[E, W]()
	for i := range z {
		z[i] = concatShiftLeft(x[i], y[i], uint64(shift[i]))
	}
	return z
}

// ShiftRightConcatModN shifts x[i] right by shift[i]%{{.xN}}, filling the
// emptied high bits with the low bits of y[i]. That is, it shifts the
// double-width concatenation of y[i] (high) and x[i] (low) right and keeps the
// low {{.xN}} bits.
//
//	s[i] = shift[i] % {{.xN}}
//	z[i] = x[i]>>s[i] | y[i]<<({{.xN}}-s[i])
//
//specgen:name ShiftRightConcatMod{{.xN}}
//specgen:require shift=Uint{xN}x{xL}
//specgen:experiment
func ShiftRightConcatModN[E Uints, W Width, sE Uints](x, y Vec[E, W], shift Vec[sE, W]) (z Vec[E, W]) {
	z = makeVec[E, W]()
	for i := range z {
		z[i] = concatShiftRight(x[i], y[i], uint64(shift[i]))
	}
	return z
}

// concatShiftLeft returns the high half of concat(x, y) << (shift % n), where n
// is the bit width of E.
func concatShiftLeft[E Uints](x, y E, shift uint64) E {
	n := uint64(elemBits[E]())
	s := shift % n
	// When s is 0, y >> n is 0.
	return E(uint64(x)<<s | uint64(y)>>(n-s))
}

// concatShiftRight returns the low half of concat(y, x) >> (shift % n), where n
// is the bit width of E.
func concatShiftRight[E Uints](x, y E, shift uint64) E {
	n := uint64(elemBits[E]())
	s := shift % n
	// When s is 0, y << n lands at bit n or above, which the conversion to E
	// discards.
	return E(uint64(x)>>s | uint64(y)<<(n-s))
}
