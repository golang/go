// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package archsimd

import "unsafe"

// LoadUint32x4Part loads a Int32x4 from the slice s.
// If s has fewer than 4 elements, the remaining elements of the vector are filled with zeroes.
// If s has 4 or more elements, the function is equivalent to LoadUint32x4.
func LoadUint32x4Part(s []uint32) (Uint32x4, int) {
	l := len(s)
	if l >= 4 {
		return LoadUint32x4(s), 4
	}
	var x Uint32x4
	if l == 0 {
		return x, 0
	}
	if l >= 2 { // 2,3
		x = x.ReshapeToUint64s().SetElem(0, *uint64atP32(&s[0])).ReshapeToUint32s()
		if l == 3 {
			x = x.SetElem(2, s[2])
		}
	} else { // l == 1
		x = x.SetElem(0, s[0])
	}
	return x, l
}

// StorePart stores the elements of x into the slice s.
// It stores as many elements as will fit in s.
// If s has 4 or more elements, the method is equivalent to x.Store.
func (x Uint32x4) StorePart(s []uint32) {
	l := len(s)
	if l >= 4 {
		x.Store(s)
		return
	}
	if l == 0 {
		return
	}
	if l >= 2 { // 2,3
		*uint64atP32(&s[0]) = x.ReshapeToUint64s().GetElem(0)
		if l == 3 {
			s[2] = x.GetElem(2)
		}
	} else { // l == 1
		s[0] = x.GetElem(0)
	}
	return
}

// LoadUint64x2Part loads a Int64x2 from the slice s.
// If s has fewer than 2 elements, the remaining elements of the vector are filled with zeroes.
// If s has 2 or more elements, the function is equivalent to LoadUint64x2.
func LoadUint64x2Part(s []uint64) (Uint64x2, int) {
	l := len(s)
	if l >= 2 {
		return LoadUint64x2(s), 2
	}
	var x Uint64x2
	if l == 0 {
		return x, 0
	}
	// l == 1
	x = x.SetElem(0, s[0])
	return x, 1
}

// StorePart stores the elements of x into the slice s.
// It stores as many elements as will fit in s.
// If s has 2 or more elements, the method is equivalent to x.Store.
func (x Uint64x2) StorePart(s []uint64) {
	l := len(s)
	if l >= 2 {
		x.Store(s)
		return
	}
	if l == 0 {
		return
	}
	// l == 1
	s[0] = x.GetElem(0)
	return
}

// LoadInt32x4Part loads a Int32x4 from the slice s.
// If s has fewer than 4 elements, the remaining elements of the vector are filled with zeroes.
// If s has 4 or more elements, the function is equivalent to LoadInt32x4.
func LoadInt32x4Part(s []int32) (Int32x4, int) {
	if len(s) == 0 {
		var zero Int32x4
		return zero, 0
	}
	t := unsafe.Slice((*uint32)(unsafe.Pointer(&s[0])), len(s))
	v, l := LoadUint32x4Part(t)
	return v.BitsToInt32(), l
}

// StorePart stores the 4 elements of x into the slice s.
// It stores as many elements as will fit in s.
// If s has 4 or more elements, the method is equivalent to x.Store.
func (x Int32x4) StorePart(s []int32) {
	if len(s) == 0 {
		return
	}
	t := unsafe.Slice((*uint32)(unsafe.Pointer(&s[0])), len(s))
	x.ToBits().StorePart(t)
}

// LoadInt64x2Part loads a Int64x2 from the slice s.
// If s has fewer than 2 elements, the remaining elements of the vector are filled with zeroes.
// If s has 2 or more elements, the function is equivalent to LoadInt64x2.
func LoadInt64x2Part(s []int64) (Int64x2, int) {
	if len(s) == 0 {
		var zero Int64x2
		return zero, 0
	}
	t := unsafe.Slice((*uint64)(unsafe.Pointer(&s[0])), len(s))
	v, l := LoadUint64x2Part(t)
	return v.BitsToInt64(), l
}

// StorePart stores the 2 elements of x into the slice s.
// It stores as many elements as will fit in s.
// If s has 2 or more elements, the method is equivalent to x.Store.
func (x Int64x2) StorePart(s []int64) {
	if len(s) == 0 {
		return
	}
	t := unsafe.Slice((*uint64)(unsafe.Pointer(&s[0])), len(s))
	x.ToBits().StorePart(t)
}

// LoadFloat32x4Part loads a Float32x4 from the slice s.
// If s has fewer than 4 elements, the remaining elements of the vector are filled with zeroes.
// If s has 4 or more elements, the function is equivalent to LoadFloat32x4.
func LoadFloat32x4Part(s []float32) (Float32x4, int) {
	l := len(s)
	if l >= 4 {
		return LoadFloat32x4(s), 4
	}
	var x Float32x4
	if l == 0 {
		return x, l
	}
	if l >= 2 { // 2,3
		x = x.ToBits().ReshapeToUint64s().BitsToFloat64().SetElem(0, *float64atP32(&s[0])).ToBits().ReshapeToUint32s().BitsToFloat32()
		if l == 3 {
			x = x.SetElem(2, s[2])
		}
	} else { // l == 1
		x = x.SetElem(0, s[0])
	}
	return x, l
}

// StorePart stores the elements of x into the slice s.
// It stores as many elements as will fit in s.
// If s has 4 or more elements, the method is equivalent to x.Store.
func (x Float32x4) StorePart(s []float32) {
	l := len(s)
	if l >= 4 {
		x.Store(s)
		return
	}
	if l == 0 {
		return
	}
	if l >= 2 { // 2,3(
		*float64atP32(&s[0]) = x.ToBits().ReshapeToUint64s().BitsToFloat64().GetElem(0)
		if l == 3 {
			s[2] = x.GetElem(2)
		}
	} else { // l == 1
		s[0] = x.GetElem(0)
	}
	return
}

// LoadFloat64x2Part loads a Float64x2 from the slice s.
// If s has fewer than 2 elements, the remaining elements of the vector are filled with zeroes.
// If s has 2 or more elements, the function is equivalent to LoadFloat64x2.
func LoadFloat64x2Part(s []float64) (Float64x2, int) {
	l := len(s)
	if l >= 2 {
		return LoadFloat64x2(s), 2
	}
	var x Float64x2
	if l == 0 {
		return x, l
	}
	// l == 1
	x = x.SetElem(0, s[0])
	return x, l
}

// StorePart stores the elements of x into the slice s.
// It stores as many elements as will fit in s.
// If s has 2 or more elements, the method is equivalent to x.Store.
func (x Float64x2) StorePart(s []float64) {
	l := len(s)
	if l >= 2 {
		x.Store(s)
		return
	}
	if l == 0 {
		return
	}
	// l == 1
	s[0] = x.GetElem(0)
	return
}
