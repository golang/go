// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package oserror defines errors values used in the os package.
//
// These types are defined here to permit the syscall package to reference them.
package oserror

import "errors"

var (
	ErrInvalid    = errors.New("invalid argument")
	ErrPermission = errors.New("permission denied")
	ErrExist      = errors.New("file already exists")
	ErrNotExist   = errors.New("file does not exist")
	ErrClosed     = errors.New("file already closed")
	ErrTimeout    = timeoutError{}
)

type timeoutError struct{}

func (timeoutError) Error() string { return "deadline exceeded" }
func (timeoutError) Timeout() bool { return true }

type temporaryError struct{}

func (temporaryError) Error() string   { return "temporary error" }
func (temporaryError) Temporary() bool { return true }

// IsTimeout reports whether err indicates a timeout.
func IsTimeout(err error) bool {
	for err != nil {
		if err == ErrTimeout {
			return true
		}
		if x, ok := err.(interface{ Timeout() bool }); ok {
			return x.Timeout()
		}
		if x, ok := err.(interface{ Is(error) bool }); ok && x.Is(ErrTimeout) {
			return true
		}
		err = errors.Unwrap(err)
	}
	return false
}
