// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd

import (
	"internal/ftoa"
	"internal/itoa"
)

type number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr | ~float32 | ~float64
}

func sliceToString[T number](x []T) string {
	s := ""
	pfx := "{"
	for _, y := range x {
		s += pfx
		pfx = ","
		switch e := any(y).(type) {
		case int8:
			s += itoa.Itoa(int(e))
		case int16:
			s += itoa.Itoa(int(e))
		case int32:
			s += itoa.Itoa(int(e))
		case int64:
			s += itoa.Itoa(int(e))
		case uint8:
			s += itoa.Uitoa(uint(e))
		case uint16:
			s += itoa.Uitoa(uint(e))
		case uint32:
			s += itoa.Uitoa(uint(e))
		case uint64:
			s += itoa.Uitoa(uint(e))
		case float32:
			s += ftoa.FormatFloat(float64(e), 'g', -1, 32)
		case float64:
			s += ftoa.FormatFloat(e, 'g', -1, 64)
		}
	}
	s += "}"
	return s
}
