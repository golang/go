// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"errors"
	"syscall"
)

func init() {
	// constants copied from GOROOT/src/internal/syscall/windows/syscall_windows.go
	const (
		ERROR_SHARING_VIOLATION syscall.Errno = 32
		ERROR_LOCK_VIOLATION    syscall.Errno = 33
	)

	isWindowsErrLockViolation = func(err error) bool {
		return errors.Is(err, ERROR_LOCK_VIOLATION)
	}

	// Copied from GOROOT/src/testing/testing_windows.go
	isWindowsRetryable = func(err error) bool {
		for {
			unwrapped := errors.Unwrap(err)
			if unwrapped == nil {
				break
			}
			err = unwrapped
		}
		if err == syscall.ERROR_ACCESS_DENIED {
			return true // Observed in https://go.dev/issue/50051.
		}
		if err == ERROR_SHARING_VIOLATION {
			return true // Observed in https://go.dev/issue/51442.
		}
		return false
	}
}
