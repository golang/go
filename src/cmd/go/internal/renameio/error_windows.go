// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package renameio

import (
	"os"
	"syscall"
)

// isAccessDeniedError returns true if err was caused by ERROR_ACCESS_DENIED.
func isAccessDeniedError(err error) bool {
	linkerr, ok := err.(*os.LinkError)
	if !ok {
		return false
	}
	errno, ok := linkerr.Err.(syscall.Errno)
	if !ok {
		return false
	}
	return errno == syscall.ERROR_ACCESS_DENIED
}
