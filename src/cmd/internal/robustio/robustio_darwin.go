// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package robustio

import (
	"errors"
	"syscall"
)

const errFileNotFound = syscall.ENOENT

// isEphemeralError returns true if err may be resolved by waiting.
func isEphemeralError(err error) bool {
	errno, ok := errors.AsType[syscall.Errno](err)
	return ok && errno == errFileNotFound
}
