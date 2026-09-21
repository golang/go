// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !windows

package os_test

import (
	"errors"
)

// isOSSymlinkUnsupportedError returns true when err is an error
// returned by os.Symlink when symlinks are unsupported by OS.
func isOSSymlinkUnsupportedError(err error) bool {
	return errors.Is(err, errors.ErrUnsupported)
}
