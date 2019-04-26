// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package errors

// A Formatter formats error messages.
type Formatter interface {
	error

	// FormatError prints the receiver's first error and returns the next error in
	// the error chain, if any.
	FormatError(p Printer) (next error)
}

// A Printer formats error messages.
//
// The most common implementation of Printer is the one provided by package fmt
// during Printf. Localization packages such as golang.org/x/text/message
// typically provide their own implementations.
type Printer interface {
	// Print appends args to the message output.
	Print(args ...interface{})

	// Printf writes a formatted string.
	Printf(format string, args ...interface{})

	// Detail reports whether error detail is requested.
	// After the first call to Detail, all text written to the Printer
	// is formatted as additional detail, or ignored when
	// detail has not been requested.
	// If Detail returns false, the caller can avoid printing the detail at all.
	Detail() bool
}
