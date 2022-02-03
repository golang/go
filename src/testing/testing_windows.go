// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package testing

import (
	"errors"
	"syscall"
)

// isWindowsAccessDenied reports whether err is ERROR_ACCESS_DENIED,
// which is defined only on Windows.
func isWindowsAccessDenied(err error) bool {
	return errors.Is(err, syscall.ERROR_ACCESS_DENIED)
}
