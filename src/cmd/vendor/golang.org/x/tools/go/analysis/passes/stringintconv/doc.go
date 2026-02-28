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
// For conversions that intend on using the code point, consider replacing them
// with string(rune(x)). Otherwise, strconv.Itoa and its equivalents return the
// string representation of the value in the desired base.
package stringintconv
