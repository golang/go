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
// The check applies to calls of the formatting functions such as
// [fmt.Printf] and [fmt.Sprintf], as well as any detected wrappers of
// those functions such as [log.Printf]. It reports a variety of
// mistakes such as syntax errors in the format string and mismatches
// (of number and type) between the verbs and their arguments.
//
// See the documentation of the fmt package for the complete set of
// format operators and their operand types.
//
// # Examples
//
// The %d format operator requires an integer operand.
// Here it is incorrectly applied to a string:
//
//	fmt.Printf("%d", "hello") // fmt.Printf format %d has arg "hello" of wrong type string
//
// A call to Printf must have as many operands as there are "verbs" in
// the format string, not too few:
//
//	fmt.Printf("%d") // fmt.Printf format reads arg 1, but call has 0 args
//
// nor too many:
//
//	fmt.Printf("%d", 1, 2) // fmt.Printf call needs 1 arg, but has 2 args
//
// Explicit argument indexes must be no greater than the number of
// arguments:
//
//	fmt.Printf("%[3]d", 1, 2) // fmt.Printf call has invalid argument index 3
//
// The checker also uses a heuristic to report calls to Print-like
// functions that appear to have been intended for their Printf-like
// counterpart:
//
//	log.Print("%d", 123) // log.Print call has possible formatting directive %d
//
// # Inferred printf wrappers
//
// Functions that delegate their arguments to fmt.Printf are
// considered "printf wrappers"; calls to them are subject to the same
// checking. In this example, logf is a printf wrapper:
//
//	func logf(level int, format string, args ...any) {
//		if enabled(level) {
//			log.Printf(format, args...)
//		}
//	}
//
//	logf(3, "invalid request: %v") // logf format reads arg 1, but call has 0 args
//
// To enable printf checking on a function that is not found by this
// analyzer's heuristics (for example, because control is obscured by
// dynamic method calls), insert a bogus call:
//
//	func MyPrintf(format string, args ...any) {
//		if false {
//			_ = fmt.Sprintf(format, args...) // enable printf checking
//		}
//		...
//	}
//
// # Specifying printf wrappers by flag
//
// The -funcs flag specifies a comma-separated list of names of
// additional known formatting functions or methods. (This legacy flag
// is rarely used due to the automatic inference described above.)
//
// If the name contains a period, it must denote a specific function
// using one of the following forms:
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
