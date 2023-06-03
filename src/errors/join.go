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
	n := 0
	for _, err := range errs {
		if err != nil {
			n++
		}
	}
	if n == 0 {
		return nil
	}
	e := &joinError{
		errs: make([]error, 0, n),
	}
	for _, err := range errs {
		if err != nil {
			e.errs = append(e.errs, err)
		}
	}
	return e
}

type joinError struct {
	errs []error
}

func (e *joinError) Error() string {
	nerrs := len(e.errs)
	// Since Join returns nil if every value in errs is nil,
	// e.errs cannot be empty.
	// TODO: get rid of case 0
	switch nerrs {
	case 0: // Impossible but handle.
		return "<nil>"
	case 1:
		return e.errs[0].Error()
	}

	const maxInt = int(^uint(0) >> 1)

	n := nerrs - 1
	for _, err := range e.errs {
		nstr := len(err.Error())
		if nstr > maxInt-n {
			panic("errors: Join output length overflow")
		}
		n += nstr
	}

	b := make([]byte, 0, n)
	b = append(b, e.errs[0].Error()...)
	for _, err := range e.errs[1:] {
		b = append(b, '\n')
		b = append(b, err.Error()...)
	}
	// At this point, b has at least one byte '\n'.
	// TODO: replace with unsafe.String(&b[0], len(b))
	return unsafe.String(unsafe.SliceData(b), len(b))
}

func (e *joinError) Unwrap() []error {
	return e.errs
}
