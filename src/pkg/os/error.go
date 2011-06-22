// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

// An Error can represent any printable error condition.
type Error interface {
	String() string
}

// // errorString is a helper type used by NewError.
type errorString string

func (e errorString) String() string { return string(e) }

// Note: If the name of the function NewError changes,
// pkg/go/doc/doc.go should be adjusted since it hardwires
// this name in a heuristic.

// // NewError returns a new error with error.String() == s.
func NewError(s string) Error { return errorString(s) }

// PathError records an error and the operation and file path that caused it.
type PathError struct {
	Op    string
	Path  string
	Error Error
}

func (e *PathError) String() string { return e.Op + " " + e.Path + ": " + e.Error.String() }
