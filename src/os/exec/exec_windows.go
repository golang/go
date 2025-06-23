// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec

import (
	"io/fs"
	"syscall"
)

// skipStdinCopyError optionally specifies a function which reports
// whether the provided stdin copy error should be ignored.
func skipStdinCopyError(err error) bool {
	// Ignore ERROR_BROKEN_PIPE and ERROR_NO_DATA errors copying
	// to stdin if the program completed successfully otherwise.
	// See Issue 20445.
	const _ERROR_NO_DATA = syscall.Errno(0xe8)
	pe, ok := err.(*fs.PathError)
	return ok &&
		pe.Op == "write" && pe.Path == "|1" &&
		(pe.Err == syscall.ERROR_BROKEN_PIPE || pe.Err == _ERROR_NO_DATA)
}
