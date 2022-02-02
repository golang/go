// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"syscall"

	errors "golang.org/x/xerrors"
)

func init() {
	// from https://docs.microsoft.com/en-us/windows/win32/debug/system-error-codes--0-499-
	const ERROR_LOCK_VIOLATION syscall.Errno = 33

	isWindowsErrLockViolation = func(err error) bool {
		return errors.Is(err, ERROR_LOCK_VIOLATION)
	}
}
