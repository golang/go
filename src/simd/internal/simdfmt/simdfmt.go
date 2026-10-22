// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package simdfmt contains formatting helpers shared by the simd packages.
package simdfmt

import "internal/strconv"

// Number is the set of element types that SliceToString can format.
type Number interface {
	int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float32 | float64
}

// SliceToString returns a string representation of x of the form "{a,b,c}".
func SliceToString[T Number](x []T) string {
	var s []byte
	s = append(s, '{')
	for i, y := range x {
		if i > 0 {
			s = append(s, ',')
		}
		switch e := any(y).(type) {
		case int8:
			s = strconv.AppendInt(s, int64(e), 10)
		case int16:
			s = strconv.AppendInt(s, int64(e), 10)
		case int32:
			s = strconv.AppendInt(s, int64(e), 10)
		case int64:
			s = strconv.AppendInt(s, e, 10)
		case uint8:
			s = strconv.AppendUint(s, uint64(e), 10)
		case uint16:
			s = strconv.AppendUint(s, uint64(e), 10)
		case uint32:
			s = strconv.AppendUint(s, uint64(e), 10)
		case uint64:
			s = strconv.AppendUint(s, e, 10)
		case float32:
			s = strconv.AppendFloat(s, float64(e), 'g', -1, 32)
		case float64:
			s = strconv.AppendFloat(s, e, 'g', -1, 64)
		}
	}
	s = append(s, '}')
	return string(s)
}
