// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A hook to get correct floating point conversion from strconv
// in packages that cannot import strconv.

package ftoa

var formatFloatPtr func(f float64, fmt byte, prec, bitSize int) string

func FormatFloat(f float64, fmt byte, prec, bitSize int) string {
	if formatFloatPtr != nil {
		return formatFloatPtr(f, fmt, prec, bitSize)
	}
	return "internal/ftoa.formatFloatPtr called before strconv.init()"
}

func SetFormatFloat(ff func(f float64, fmt byte, prec, bitSize int) string) {
	if formatFloatPtr == nil {
		formatFloatPtr = ff
	}
}
