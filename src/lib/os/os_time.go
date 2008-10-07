// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"os";
	"syscall"
)

export func Time() (sec int64, nsec int64, err *Error) {
	var errno int64;
	sec, nsec, errno = syscall.gettimeofday();
	if errno != 0 {
		return 0, 0, ErrnoToError(errno)
	}
	return sec, nsec, nil
}

