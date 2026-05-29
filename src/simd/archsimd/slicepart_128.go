// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || arm64 || wasm)

package archsimd

import "unsafe"

// Implementation of all the {Int,Uint}{8,16} load and store part
// functions and methods for 128-bit for architectures that must do that by pieces.

/* pointer-punning functions for chunked part-of-slice loads. */

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

func uint16atP8(p *uint8) *uint16 {
	return (*uint16)(unsafe.Pointer(p))
}

func uint32atP8(p *uint8) *uint32 {
	return (*uint32)(unsafe.Pointer(p))
}

func uint64atP8(p *uint8) *uint64 {
	return (*uint64)(unsafe.Pointer(p))
}

func uint32atP16(p *uint16) *uint32 {
	return (*uint32)(unsafe.Pointer(p))
}

func uint64atP16(p *uint16) *uint64 {
	return (*uint64)(unsafe.Pointer(p))
}

func uint64atP32(p *uint32) *uint64 {
	return (*uint64)(unsafe.Pointer(p))
}

func uint32atP64(p *uint64) *uint32 {
	return (*uint32)(unsafe.Pointer(p))
}

func float64atP32(p *float32) *float64 {
	return (*float64)(unsafe.Pointer(p))
}

func float32atP64(p *float64) *float32 {
	return (*float32)(unsafe.Pointer(p))
}

/* 128-bit vector load and store slice parts for 8 and 16-bit int elements */

// LoadUint8x16Part loads a Uint8x16 from the slice s.
// If s has fewer than 16 elements, the remaining elements of the vector are filled with zeroes.
// If s has 16 or more elements, the function is equivalent to LoadInt8x16.
func LoadUint8x16Part(s []uint8) (Uint8x16, int) {
	l := len(s)
	if l >= 16 {
		return LoadUint8x16(s), 16
	}
	var x Uint8x16
	if l == 0 {
		return x, 0
	}
	if l >= 8 { // 8-15
		x = x.ReshapeToUint64s().SetElem(0, *uint64atP8(&s[0])).ReshapeToUint8s()
		if l >= 12 { // 12, 13, 14, 15
			x = x.ReshapeToUint32s().SetElem(8/4, *uint32atP8(&s[8])).ReshapeToUint8s()
			if l >= 14 {
				x = x.ReshapeToUint16s().SetElem(12/2, *uint16atP8(&s[12])).ReshapeToUint8s()
				if l == 15 {
					x = x.SetElem(14, s[14])
				}
			} else if l == 13 {
				x = x.SetElem(12, s[12])
			}
		} else if l >= 10 { // 10, 11
			x = x.ReshapeToUint16s().SetElem(8/2, *uint16atP8(&s[8])).ReshapeToUint8s()
			if l == 11 {
				x = x.SetElem(10, s[10])
			}
		} else if l == 9 {
			x = x.SetElem(8, s[8])
		}
	} else if l >= 4 { // 4-7
		x = x.ReshapeToUint32s().SetElem(0, *uint32atP8(&s[0])).ReshapeToUint8s()
		if l >= 6 {
			x = x.ReshapeToUint16s().SetElem(4/2, *uint16atP8(&s[4])).ReshapeToUint8s()
			if l == 7 {
				x = x.SetElem(6, s[6])
			}
		} else if l == 5 {
			x = x.SetElem(4, s[4])
		}
	} else if l >= 2 { // 2,3
		x = x.ReshapeToUint16s().SetElem(0, *uint16atP8(&s[0])).ReshapeToUint8s()
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
// If s has 16 or more elements, the method is equivalent to x.Store.
func (x Uint8x16) StorePart(s []uint8) {
	l := len(s)
	if l >= 16 {
		x.Store(s)
		return
	}
	if l == 0 {
		return
	}
	if l >= 8 { // 8-15
		*uint64atP8(&s[0]) = x.ReshapeToUint64s().GetElem(0)
		if l >= 12 { // 12, 13, 14, 15
			*uint32atP8(&s[8]) = x.ReshapeToUint32s().GetElem(8 / 4)
			if l >= 14 {
				*uint16atP8(&s[12]) = x.ReshapeToUint16s().GetElem(12 / 2)
				if l == 15 {
					s[14] = x.GetElem(14)
				}
			} else if l == 13 {
				s[12] = x.GetElem(12)
			}
		} else if l >= 10 { // 10, 11
			*uint16atP8(&s[8]) = x.ReshapeToUint16s().GetElem(8 / 2)
			if l == 11 {
				s[10] = x.GetElem(10)
			}
		} else if l == 9 {
			s[8] = x.GetElem(8)
		}
	} else if l >= 4 { // 4-7
		*uint32atP8(&s[0]) = x.ReshapeToUint32s().GetElem(0)
		if l >= 6 {
			*uint16atP8(&s[4]) = x.ReshapeToUint16s().GetElem(4 / 2)
			if l == 7 {
				s[6] = x.GetElem(6)
			}
		} else if l == 5 {
			s[4] = x.GetElem(4)
		}
	} else if l >= 2 { // 2,3
		*uint16atP8(&s[0]) = x.ReshapeToUint16s().GetElem(0)
		if l == 3 {
			s[2] = x.GetElem(2)
		}
	} else { // l == 1
		s[0] = x.GetElem(0)
	}
}

// LoadUint16x8Part loads a Uint16x8 from the slice s.
// If s has fewer than 8 elements, the remaining elements of the vector are filled with zeroes.
// If s has 8 or more elements, the function is equivalent to LoadInt16x8.
func LoadUint16x8Part(s []uint16) (Uint16x8, int) {
	l := len(s)
	if l >= 8 {
		return LoadUint16x8(s), 8
	}
	var x Uint16x8
	if l == 0 {
		return x, 0
	}
	if l >= 4 { // 4-7
		x = x.ReshapeToUint64s().SetElem(0, *uint64atP16(&s[0])).ReshapeToUint16s()
		if l >= 6 {
			x = x.ReshapeToUint32s().SetElem(4/2, *uint32atP16(&s[4])).ReshapeToUint16s()
			if l == 7 {
				x = x.SetElem(6, s[6])
			}
		} else if l == 5 {
			x = x.SetElem(4, s[4])
		}
	} else if l >= 2 { // 2,3
		x = x.ReshapeToUint32s().SetElem(0, *uint32atP16(&s[0])).ReshapeToUint16s()
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
// If s has 8 or more elements, the method is equivalent to x.Store.
func (x Uint16x8) StorePart(s []uint16) {
	l := len(s)
	if l >= 8 {
		x.Store(s)
		return
	}
	if l == 0 {
		return
	}
	if l >= 4 { // 4-7
		*uint64atP16(&s[0]) = x.ReshapeToUint64s().GetElem(0)
		if l >= 6 {
			*uint32atP16(&s[4]) = x.ReshapeToUint32s().GetElem(4 / 2)
			if l == 7 {
				s[6] = x.GetElem(6)
			}
		} else if l == 5 {
			s[4] = x.GetElem(4)
		}
	} else if l >= 2 { // 2,3
		*uint32atP16(&s[0]) = x.ReshapeToUint32s().GetElem(0)
		if l == 3 {
			s[2] = x.GetElem(2)
		}
	} else { // l == 1
		s[0] = x.GetElem(0)
	}
	return
}

// LoadInt8x16Part loads a Int8x16 from the slice s, it returns the loaded vector and the
// number of elements loaded.
// If s has fewer than 16 elements, the remaining elements of the vector are filled with zeroes.
// If s has 16 or more elements, the function is equivalent to LoadInt8x16.
func LoadInt8x16Part(s []int8) (Int8x16, int) {
	if len(s) == 0 {
		var zero Int8x16
		return zero, 0
	}
	t := unsafe.Slice((*uint8)(unsafe.Pointer(&s[0])), len(s))
	v, l := LoadUint8x16Part(t)
	return v.BitsToInt8(), l
}

// StorePart stores the 16 elements of x into the slice s.
// It stores as many elements as will fit in s.
// If s has 16 or more elements, the method is equivalent to x.Store.
func (x Int8x16) StorePart(s []int8) {
	if len(s) == 0 {
		return
	}
	t := unsafe.Slice((*uint8)(unsafe.Pointer(&s[0])), len(s))
	x.ToBits().StorePart(t)
}

// LoadInt16x8Part loads a Int16x8 from the slice s, it returns the loaded vector and the
// number of elements loaded.
// If s has fewer than 8 elements, the remaining elements of the vector are filled with zeroes.
// If s has 8 or more elements, the function is equivalent to LoadInt16x8.
func LoadInt16x8Part(s []int16) (Int16x8, int) {
	if len(s) == 0 {
		var zero Int16x8
		return zero, 0
	}
	t := unsafe.Slice((*uint16)(unsafe.Pointer(&s[0])), len(s))
	v, l := LoadUint16x8Part(t)
	return v.BitsToInt16(), l
}

// StorePart stores the 8 elements of x into the slice s.
// It stores as many elements as will fit in s.
// If s has 8 or more elements, the method is equivalent to x.Store.
func (x Int16x8) StorePart(s []int16) {
	if len(s) == 0 {
		return
	}
	t := unsafe.Slice((*uint16)(unsafe.Pointer(&s[0])), len(s))
	x.ToBits().StorePart(t)
}
