// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import "syscall"

var (
	errOpNotSupported = syscall.EPLAN9

	abortedConnRequestErrors []error
)

func isPlatformError(err error) bool {
	_, ok := err.(syscall.ErrorString)
	return ok
}

func isENOBUFS(err error) bool {
	return false // ENOBUFS is Unix-specific
}
