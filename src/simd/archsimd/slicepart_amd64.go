// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package archsimd

import "unsafe"

// Implementation of all the {Int,Uint}{8,16} load and store slice part
// functions and methods for 128-bit and 256-bit vectors.

/* pointer-punning functions for chunked slice part loads. */

func int16atP8(p *int8) *int16 {
	return (*int16)(unsafe.Pointer(p))
}

func int32atP8(p *int8) *int32 {
	return (*int32)(unsafe.Pointer(p))
}

func int64atP8(p *int8) *int64 {
	return (*int64)(unsafe.Pointer(p))
}

func int32atP16(p *int16) *int32 {
	return (*int32)(unsafe.Pointer(p))
}

func int64atP16(p *int16) *int64 {
	return (*int64)(unsafe.Pointer(p))
}

func int64atP32(p *int32) *int64 {
	return (*int64)(unsafe.Pointer(p))
}

func int32atP64(p *int64) *int32 {
	return (*int32)(unsafe.Pointer(p))
}

/* These two masks are used by generated code */

var vecMask64 = [16]int64{
	-1, -1, -1, -1,
	-1, -1, -1, -1,
	0, 0, 0, 0,
	0, 0, 0, 0,
}

var vecMask32 = [32]int32{
	-1, -1, -1, -1,
	-1, -1, -1, -1,
	-1, -1, -1, -1,
	-1, -1, -1, -1,
	0, 0, 0, 0,
	0, 0, 0, 0,
	0, 0, 0, 0,
	0, 0, 0, 0,
}

/* 256-bit int vector loads and stores made from 128-bit parts */

// LoadInt8x32SlicePart loads a Int8x32 from the slice s.
// If s has fewer than 32 elements, the remaining elements of the vector are filled with zeroes.
// If s has 32 or more elements, the function is equivalent to LoadInt8x32Slice.
func LoadInt8x32SlicePart(s []int8) Int8x32 {
	l := len(s)
	if l >= 32 {
		return LoadInt8x32Slice(s)
	}
	var x Int8x32
	if l == 0 {
		return x
	}
	if l > 16 {
		return x.SetLo(LoadInt8x16Slice(s)).SetHi(LoadInt8x16SlicePart(s[16:]))
	} else {
		return x.SetLo(LoadInt8x16SlicePart(s))
	}
}

// LoadInt16x16SlicePart loads a Int16x16 from the slice s.
// If s has fewer than 16 elements, the remaining elements of the vector are filled with zeroes.
// If s has 16 or more elements, the function is equivalent to LoadInt16x16Slice.
func LoadInt16x16SlicePart(s []int16) Int16x16 {
	l := len(s)
	if l >= 16 {
		return LoadInt16x16Slice(s)
	}
	var x Int16x16
	if l == 0 {
		return x
	}
	if l > 8 {
		return x.SetLo(LoadInt16x8Slice(s)).SetHi(LoadInt16x8SlicePart(s[8:]))
	} else {
		return x.SetLo(LoadInt16x8SlicePart(s))
	}
}

// StoreSlicePart stores the elements of x into the slice s.
// It stores as many elements as will fit in s.
// If s has 32 or more elements, the method is equivalent to x.StoreSlice.
func (x Int8x32) StoreSlicePart(s []int8) {
	l := len(s)
	if l >= 32 {
		x.StoreSlice(s)
		return
	}
	if l == 0 {
		return
	}
	if l > 16 {
		x.GetLo().StoreSlice(s)
		x.GetHi().StoreSlicePart(s[16:])
	} else { // fits in one
		x.GetLo().StoreSlicePart(s)
	}
}

// StoreSlicePart stores the elements of x into the slice s.
// It stores as many elements as will fit in s.
// If s has 16 or more elements, the method is equivalent to x.StoreSlice.
func (x Int16x16) StoreSlicePart(s []int16) {
	l := len(s)
	if l >= 16 {
		x.StoreSlice(s)
		return
	}
	if l == 0 {
		return
	}
	if l > 8 {
		x.GetLo().StoreSlice(s)
		x.GetHi().StoreSlicePart(s[8:])
	} else { // fits in one
		x.GetLo().StoreSlicePart(s)
	}
}

/* 128-bit vector load and store slice parts for 8 and 16-bit int elements */

// LoadInt8x16SlicePart loads a Int8x16 from the slice s.
// If s has fewer than 16 elements, the remaining elements of the vector are filled with zeroes.
// If s has 16 or more elements, the function is equivalent to LoadInt8x16Slice.
func LoadInt8x16SlicePart(s []int8) Int8x16 {
	l := len(s)
	if l >= 16 {
		return LoadInt8x16Slice(s)
	}
	var x Int8x16
	if l == 0 {
		return x
	}
	if l >= 8 { // 8-15
		x = x.AsInt64x2().SetElem(0, *int64atP8(&s[0])).AsInt8x16()
		if l >= 12 { // 12, 13, 14, 15
			x = x.AsInt32x4().SetElem(8/4, *int32atP8(&s[8])).AsInt8x16()
			if l >= 14 {
				x = x.AsInt16x8().SetElem(12/2, *int16atP8(&s[12])).AsInt8x16()
				if l == 15 {
					x = x.SetElem(14, s[14])
				}
			} else if l == 13 {
				x = x.SetElem(12, s[12])
			}
		} else if l >= 10 { // 10, 11
			x = x.AsInt16x8().SetElem(8/2, *int16atP8(&s[8])).AsInt8x16()
			if l == 11 {
				x = x.SetElem(10, s[10])
			}
		} else if l == 9 {
			x = x.SetElem(8, s[8])
		}
	} else if l >= 4 { // 4-7
		x = x.AsInt32x4().SetElem(0, *int32atP8(&s[0])).AsInt8x16()
		if l >= 6 {
			x = x.AsInt16x8().SetElem(4/2, *int16atP8(&s[4])).AsInt8x16()
			if l == 7 {
				x = x.SetElem(6, s[6])
			}
		} else if l == 5 {
			x = x.SetElem(4, s[4])
		}
	} else if l >= 2 { // 2,3
		x = x.AsInt16x8().SetElem(0, *int16atP8(&s[0])).AsInt8x16()
		if l == 3 {
			x = x.SetElem(2, s[2])
		}
	} else { // l == 1
		x = x.SetElem(0, s[0])
	}
	return x
}

// StoreSlicePart stores the elements of x into the slice s.
// It stores as many elements as will fit in s.
// If s has 16 or more elements, the method is equivalent to x.StoreSlice.
func (x Int8x16) StoreSlicePart(s []int8) {
	l := len(s)
	if l >= 16 {
		x.StoreSlice(s)
		return
	}
	if l == 0 {
		return
	}
	if l >= 8 { // 8-15
		*int64atP8(&s[0]) = x.AsInt64x2().GetElem(0)
		if l >= 12 { // 12, 13, 14, 15
			*int32atP8(&s[8]) = x.AsInt32x4().GetElem(8 / 4)
			if l >= 14 {
				*int16atP8(&s[12]) = x.AsInt16x8().GetElem(12 / 2)
				if l == 15 {
					s[14] = x.GetElem(14)
				}
			} else if l == 13 {
				s[12] = x.GetElem(12)
			}
		} else if l >= 10 { // 10, 11
			*int16atP8(&s[8]) = x.AsInt16x8().GetElem(8 / 2)
			if l == 11 {
				s[10] = x.GetElem(10)
			}
		} else if l == 9 {
			s[8] = x.GetElem(8)
		}
	} else if l >= 4 { // 4-7
		*int32atP8(&s[0]) = x.AsInt32x4().GetElem(0)
		if l >= 6 {
			*int16atP8(&s[4]) = x.AsInt16x8().GetElem(4 / 2)
			if l == 7 {
				s[6] = x.GetElem(6)
			}
		} else if l == 5 {
			s[4] = x.GetElem(4)
		}
	} else if l >= 2 { // 2,3
		*int16atP8(&s[0]) = x.AsInt16x8().GetElem(0)
		if l == 3 {
			s[2] = x.GetElem(2)
		}
	} else { // l == 1
		s[0] = x.GetElem(0)
	}
}

// LoadInt16x8SlicePart loads a Int16x8 from the slice s.
// If s has fewer than 8 elements, the remaining elements of the vector are filled with zeroes.
// If s has 8 or more elements, the function is equivalent to LoadInt16x8Slice.
func LoadInt16x8SlicePart(s []int16) Int16x8 {
	l := len(s)
	if l >= 8 {
		return LoadInt16x8Slice(s)
	}
	var x Int16x8
	if l == 0 {
		return x
	}
	if l >= 4 { // 4-7
		x = x.AsInt64x2().SetElem(0, *int64atP16(&s[0])).AsInt16x8()
		if l >= 6 {
			x = x.AsInt32x4().SetElem(4/2, *int32atP16(&s[4])).AsInt16x8()
			if l == 7 {
				x = x.SetElem(6, s[6])
			}
		} else if l == 5 {
			x = x.SetElem(4, s[4])
		}
	} else if l >= 2 { // 2,3
		x = x.AsInt32x4().SetElem(0, *int32atP16(&s[0])).AsInt16x8()
		if l == 3 {
			x = x.SetElem(2, s[2])
		}
	} else { // l == 1
		x = x.SetElem(0, s[0])
	}
	return x
}

// StoreSlicePart stores the elements of x into the slice s.
// It stores as many elements as will fit in s.
// If s has 8 or more elements, the method is equivalent to x.StoreSlice.
func (x Int16x8) StoreSlicePart(s []int16) {
	l := len(s)
	if l >= 8 {
		x.StoreSlice(s)
		return
	}
	if l == 0 {
		return
	}
	if l >= 4 { // 4-7
		*int64atP16(&s[0]) = x.AsInt64x2().GetElem(0)
		if l >= 6 {
			*int32atP16(&s[4]) = x.AsInt32x4().GetElem(4 / 2)
			if l == 7 {
				s[6] = x.GetElem(6)
			}
		} else if l == 5 {
			s[4] = x.GetElem(4)
		}
	} else if l >= 2 { // 2,3
		*int32atP16(&s[0]) = x.AsInt32x4().GetElem(0)
		if l == 3 {
			s[2] = x.GetElem(2)
		}
	} else { // l == 1
		s[0] = x.GetElem(0)
	}
	return
}
