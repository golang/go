// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package composite

// unkeyedLiteral is a white list of types in the standard packages
// that are used with unkeyed literals we deem to be acceptable.
var unkeyedLiteral = map[string]bool{
	// These image and image/color struct types are frozen. We will never add fields to them.
	"image/color.Alpha16": true,
	"image/color.Alpha":   true,
	"image/color.CMYK":    true,
	"image/color.Gray16":  true,
	"image/color.Gray":    true,
	"image/color.NRGBA64": true,
	"image/color.NRGBA":   true,
	"image/color.NYCbCrA": true,
	"image/color.RGBA64":  true,
	"image/color.RGBA":    true,
	"image/color.YCbCr":   true,
	"image.Point":         true,
	"image.Rectangle":     true,
	"image.Uniform":       true,

	"unicode.Range16": true,
	"unicode.Range32": true,

	// These three structs are used in generated test main files,
	// but the generator can be trusted.
	"testing.InternalBenchmark": true,
	"testing.InternalExample":   true,
	"testing.InternalTest":      true,
}
