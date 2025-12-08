// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package archsimd

import (
	"internal/strconv"
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
			s += strconv.Itoa(int(e))
		case int16:
			s += strconv.Itoa(int(e))
		case int32:
			s += strconv.Itoa(int(e))
		case int64:
			s += strconv.Itoa(int(e))
		case uint8:
			s += strconv.FormatUint(uint64(e), 10)
		case uint16:
			s += strconv.FormatUint(uint64(e), 10)
		case uint32:
			s += strconv.FormatUint(uint64(e), 10)
		case uint64:
			s += strconv.FormatUint(uint64(e), 10)
		case float32:
			s += strconv.FormatFloat(float64(e), 'g', -1, 32)
		case float64:
			s += strconv.FormatFloat(e, 'g', -1, 64)
		}
	}
	s += "}"
	return s
}
