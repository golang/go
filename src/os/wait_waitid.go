// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// We used to used this code for Darwin, but according to issue #19314
// waitid returns if the process is stopped, even when using WEXITED.

//go:build linux

package os

import (
	"internal/poll"
	"syscall"
)

const _P_PIDFD = 3

// blockUntilWaitable attempts to block until a call to p.Wait will
// succeed immediately, and reports whether it has done so.
// It does not actually call p.Wait.
func (p *Process) blockUntilWaitable() (bool, error) {
	fd := poll.FD{
		Sysfd: int(p.handle),
	}
	err := fd.Init("pidfd", false)
	if err != nil {
		return false, err
	}
	// We just want to make sure fd is ready for reading, but that is not yet available.
	// See: https://github.com/golang/go/issues/15735
	buf := make([]byte, 1)
	_, err = fd.Read(buf)
	if err == syscall.EINVAL {
		// fd is ready for reading, but reading failed, as expected on pidfd.
		return true, nil
	} else if err != nil {
		return false, err
	}
	return true, nil
}
