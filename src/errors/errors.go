// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package errors implements functions to manipulate errors.
//
// The [New] function creates errors whose only content is a text message.
//
// An error e wraps another error if e's type has one of the methods
//
//	Unwrap() error
//	Unwrap() []error
//
// If e.Unwrap() returns a non-nil error w or a slice containing w,
// then we say that e wraps w. A nil error returned from e.Unwrap()
// indicates that e does not wrap any error. It is invalid for an
// Unwrap method to return an []error containing a nil error value.
//
// An easy way to create wrapped errors is to call [fmt.Errorf] and apply
// the %w verb to the error argument:
//
//	wrapsErr := fmt.Errorf("... %w ...", ..., err, ...)
//
// Successive unwrapping of an error creates a tree. The [Is] and [As]
// functions inspect an error's tree by examining first the error
// itself followed by the tree of each of its children in turn
// (pre-order, depth-first traversal).
//
// [Is] examines the tree of its first argument looking for an error that
// matches the second. It reports whether it finds a match. It should be
// used in preference to simple equality checks:
//
//	if errors.Is(err, fs.ErrExist)
//
// is preferable to
//
//	if err == fs.ErrExist
//
// because the former will succeed if err wraps [io/fs.ErrExist].
//
// [As] examines the tree of its first argument looking for an error that can be
// assigned to its second argument, which must be a pointer. If it succeeds, it
// performs the assignment and returns true. Otherwise, it returns false. The form
//
//	var perr *fs.PathError
//	if errors.As(err, &perr) {
//		fmt.Println(perr.Path)
//	}
//
// is preferable to
//
//	if perr, ok := err.(*fs.PathError); ok {
//		fmt.Println(perr.Path)
//	}
//
// because the former will succeed if err wraps an [*io/fs.PathError].
package errors

// New returns an error that formats as the given text.
// Each call to New returns a distinct error value even if the text is identical.
func New(text string) error {
	return &errorString{text}
}

// errorString is a trivial implementation of error.
type errorString struct {
	s string
}

func (e *errorString) Error() string {
	return e.s
}

// ErrUnsupported indicates that a requested operation cannot be performed,
// because it is unsupported. For example, a call to [os.Link] when using a
// file system that does not support hard links.
//
// Functions and methods should not return this error but should instead
// return an error including appropriate context that satisfies
//
//	errors.Is(err, errors.ErrUnsupported)
//
// either by directly wrapping ErrUnsupported or by implementing an [Is] method.
//
// Functions and methods should document the cases in which an error
// wrapping this will be returned.
var ErrUnsupported = New("unsupported operation")
