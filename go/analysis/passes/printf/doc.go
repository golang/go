// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package printf defines an Analyzer that checks consistency
// of Printf format strings and arguments.
//
// # Analyzer printf
//
// printf: check consistency of Printf format strings and arguments
//
// The check applies to known functions (for example, those in package fmt)
// as well as any detected wrappers of known functions.
//
// To enable printf checking on a function that is not found by this
// analyzer's heuristics (for example, because control is obscured by
// dynamic method calls), insert a bogus call:
//
//	func MyPrintf(format string, args ...any) {
//		if false {
//			_ = fmt.Sprintf(format, args...) // enable printf checker
//		}
//		...
//	}
//
// The -funcs flag specifies a comma-separated list of names of additional
// known formatting functions or methods. If the name contains a period,
// it must denote a specific function using one of the following forms:
//
//	dir/pkg.Function
//	dir/pkg.Type.Method
//	(*dir/pkg.Type).Method
//
// Otherwise the name is interpreted as a case-insensitive unqualified
// identifier such as "errorf". Either way, if a listed name ends in f, the
// function is assumed to be Printf-like, taking a format string before the
// argument list. Otherwise it is assumed to be Print-like, taking a list
// of arguments with no format string.
package printf
