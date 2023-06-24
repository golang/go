// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netgo && netcgo

package net

func init() {
	// This will give a compile time error about the unused constant.
	// The advantage of this approach is that the gc compiler
	// actually prints the constant, making the problem obvious.
	"Do not use both netgo and netcgo build tags."
}
