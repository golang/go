// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netgo && netcgo

package net

func init() {
	// When both netgo and netcgo build tags are being used
	// at the same time, this unused string literal will
	// cause a compiler error with the contents of this string included.
	"Do not use both netgo and netcgo build tags."
}
