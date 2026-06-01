// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package variants

// DO NOT EDIT WITHOUT READING THIS FIRST.

// This file is shared by the simd bridge generator
// and the compiler's midway tree rewriter.  The sharing
// is facilitated by an automatic copy from the file
// embedded in the simd bridge generator, which itself
// copies it from its "variants" subdirectory.  To
// change this file, edit the copy in
// src/simd/archsimd/_gen/midway/variants/variants.go,
// and then rerun ("go run") the simd bridge generator
// and it will create a bridge consistent with the contents
// of this file, for use with a compiler built to use this
// file.  Minor version skews will only affect code
// (e.g., rebuilding the compiler) that uses "simd"
// (not "archsimd") and that runs on hardware lacking
// the expected feature set.  In the worst case,
// GODEBUG=simd=0 will avoid the problem, unless it
// involves missing emulation methods (which you are
// expected to write anyway if you are tinkering with
// this code).

type Key struct {
	Arch string
	Size int
}

type Variant struct {
	Suffix          string          // the name (suffix) for this variant.
	Emulated        map[string]bool // Methods that are emulated for this variant.
	DefaultRequires string          // default requires this function return true, false means variant.
}

// Name returns the variant name of type t.
// If receiver v is nil, returns t.
func (v *Variant) Name(t string) string {
	if v == nil {
		return t
	}
	return t + v.Suffix
}

var Variants map[Key]*Variant

func init() {
	Variants = make(map[Key]*Variant)
	Variants[Key{"arm64", 128}] = &Variant{
		Suffix: "nclm",
		Emulated: map[string]bool{
			"CarrylessMultiplyEven": true,
			"CarrylessMultiplyOdd":  true,
		},
		DefaultRequires: "HasHardwareCarrylessMultiply",
	}

}
