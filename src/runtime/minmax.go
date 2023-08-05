// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

func strmin(x, y string) string {
	if y < x {
		return y
	}
	return x
}

func strmax(x, y string) string {
	if y > x {
		return y
	}
	return x
}

func fmin32(x, y float32) float32 { return fmin(x, y) }
func fmin64(x, y float64) float64 { return fmin(x, y) }
func fmax32(x, y float32) float32 { return fmax(x, y) }
func fmax64(x, y float64) float64 { return fmax(x, y) }

type floaty interface{ ~float32 | ~float64 }

func fmin[F floaty](x, y F) F {
	if y != y || y < x {
		return y
	}
	if x != x || x < y || x != 0 {
		return x
	}
	// x and y are both ±0
	// if either is -0, return -0; else return +0
	return forbits(x, y)
}

func fmax[F floaty](x, y F) F {
	if y != y || y > x {
		return y
	}
	if x != x || x > y || x != 0 {
		return x
	}
	// x and y are both ±0
	// if both are -0, return -0; else return +0
	return fandbits(x, y)
}

func forbits[F floaty](x, y F) F {
	switch unsafe.Sizeof(x) {
	case 4:
		*(*uint32)(unsafe.Pointer(&x)) |= *(*uint32)(unsafe.Pointer(&y))
	case 8:
		*(*uint64)(unsafe.Pointer(&x)) |= *(*uint64)(unsafe.Pointer(&y))
	}
	return x
}

func fandbits[F floaty](x, y F) F {
	switch unsafe.Sizeof(x) {
	case 4:
		*(*uint32)(unsafe.Pointer(&x)) &= *(*uint32)(unsafe.Pointer(&y))
	case 8:
		*(*uint64)(unsafe.Pointer(&x)) &= *(*uint64)(unsafe.Pointer(&y))
	}
	return x
}
