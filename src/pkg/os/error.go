// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

// An Error can represent any printable error condition.
type Error interface {
	String() string
}

// A helper type that can be embedded or wrapped to simplify satisfying
// Error.
type ErrorString string

func (e ErrorString) String() string  { return string(e) }
func (e ErrorString) Temporary() bool { return false }
func (e ErrorString) Timeout() bool   { return false }

// Note: If the name of the function NewError changes,
// pkg/go/doc/doc.go should be adjusted since it hardwires
// this name in a heuristic.

// NewError converts s to an ErrorString, which satisfies the Error interface.
func NewError(s string) Error { return ErrorString(s) }

// PathError records an error and the operation and file path that caused it.
type PathError struct {
	Op    string
	Path  string
	Error Error
}

func (e *PathError) String() string { return e.Op + " " + e.Path + ": " + e.Error.String() }
