// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || netbsd

package syscall

// blockUntilWaitable attempts to block until a call to Wait4 will
// succeed immediately, and reports whether it has done so.
// It does not actually call Wait4.
func blockUntilWaitable(searchPID int) (int, error) {
	idType := _P_ALL
	if searchPID != 0 {
		idType = _P_PID
	}
	for {
		if pid, err := wait6(idType, searchPID, WEXITED|WNOWAIT); err == ENOSYS {
			return 0, nil
		} else if err != EINTR {
			return pid, err
		}
	}
}
