// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package errors implements functions to manipulate errors.
package errors

import "runtime"

// New returns an error that formats as the given text.
//
// The returned error contains a Frame set to the caller's location and
// implements Formatter to show this information when printed with details.
func New(text string) error {
	// Inline call to errors.Callers to improve performance.
	var s Frame
	runtime.Callers(2, s.frames[:])
	return &errorString{text, s}
}

// errorString is a trivial implementation of error.
type errorString struct {
	s     string
	frame Frame
}

func (e *errorString) Error() string {
	return e.s
}

func (e *errorString) FormatError(p Printer) (next error) {
	p.Print(e.s)
	e.frame.Format(p)
	return nil
}
