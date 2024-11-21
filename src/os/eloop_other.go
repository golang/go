// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || openbsd || solaris || wasip1

package os

import (
	"runtime"
	"syscall"
)

// isNoFollowErr reports whether err may result from O_NOFOLLOW blocking an open operation.
func isNoFollowErr(err error) bool {
	switch err {
	case syscall.ELOOP, syscall.EMLINK:
		return true
	}
	if runtime.GOOS == "dragonfly" {
		// Dragonfly appears to return EINVAL from openat in this case.
		if err == syscall.EINVAL {
			return true
		}
	}
	return false
}
