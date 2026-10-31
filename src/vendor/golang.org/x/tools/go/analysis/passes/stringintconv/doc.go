// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package stringintconv defines an Analyzer that flags type conversions
// from integers to strings.
//
// # Analyzer stringintconv
//
// stringintconv: check for string(int) conversions
//
// This checker flags conversions of the form string(x) where x is an integer
// (but not byte or rune) type. Such conversions are discouraged because they
// return the UTF-8 representation of the Unicode code point x, and not a decimal
// string representation of x as one might expect. Furthermore, if x denotes an
// invalid code point, the conversion cannot be statically rejected.
//
// As of Go 1.28, such conversions are rejected by the compiler in
// files whose Go version is go1.28 or later.
//
// The checker offers two fixes. The first, which is applied by "go
// fix", preserves the existing behavior by converting x to a rune
// first: string(rune(x)). If x may be wider than 32 bits, truncation
// to a rune could turn an invalid code point into a valid one, so
// the fix instead uses fmt.Sprintf("%c", x), which, like string(x),
// yields "\uFFFD" for all values outside the range of valid code
// points.
//
// The second fix formats the number as a decimal using
// fmt.Sprintf("%d", x), which is usually what was intended,
// but changes the behavior.
//
// To migrate a module to Go 1.28, run "go fix" before updating the
// go directive in its go.mod file, since after that, the package no
// longer compiles and cannot be analyzed.
package stringintconv
