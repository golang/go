// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package errors

import (
	"unsafe"
)

// Join returns an error that wraps the given errors.
// Any nil error values are discarded.
// Join returns nil if every value in errs is nil.
// The error formats as the concatenation of the strings obtained
// by calling the Error method of each element of errs, with a newline
// between each string.
//
// A non-nil error returned by Join implements the Unwrap() []error method.
func Join(errs ...error) error {
	allErrs := make([]error, 0, len(errs))
	for _, e := range errs {
		// Ignore nil errors.
		if e == nil {
			continue
		}

		// Specifically handle nested join errors from the standard library. This
		// avoids deeply-nesting values which can be unexpected when unwrapping
		// errors.
		joinErr, ok := e.(*joinError)
		if !ok {
			allErrs = append(allErrs, e)
			continue
		}

		allErrs = append(allErrs, joinErr.errs...)
	}

	// Ensure we return nil if all contained errors were nil.
	if len(allErrs) == 0 {
		return nil
	}

	return &joinError{
		errs: allErrs,
	}
}

type joinError struct {
	errs []error
}

func (e *joinError) Error() string {
	// Since Join returns nil if every value in errs is nil,
	// e.errs cannot be empty.
	if len(e.errs) == 1 {
		return e.errs[0].Error()
	}

	b := []byte(e.errs[0].Error())
	for _, err := range e.errs[1:] {
		b = append(b, '\n')
		b = append(b, err.Error()...)
	}
	// At this point, b has at least one byte '\n'.
	return unsafe.String(&b[0], len(b))
}

func (e *joinError) Unwrap() []error {
	return e.errs
}
