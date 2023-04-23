// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netgo && netcgo

package net

func init() {
	// This will cause a compiler error with the contents of
	// this unused string included, when both netgo and netcgo
	// build tags are being used at the same time.
	"Do not use both netgo and netcgo build tags."
}
