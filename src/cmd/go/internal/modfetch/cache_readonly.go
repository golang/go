// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9

package modfetch

import (
	"errors"
	"runtime"
	"syscall"
)

// isErrReadOnlyFS reports whether err is a read-only filesystem error
// (syscall.EROFS on Unix/wasip1, ERROR_NOT_SUPPORTED on Windows).
func isErrReadOnlyFS(err error) bool {
	switch runtime.GOOS {
	case "windows":
		const ERROR_NOT_SUPPORTED = 50
		return errors.Is(err, syscall.Errno(ERROR_NOT_SUPPORTED))
	case "wasip1":
		const EROFS = 69
		return errors.Is(err, syscall.Errno(EROFS))
	default: // unix and js
		const EROFS = 30
		return errors.Is(err, syscall.Errno(EROFS))
	}
}
