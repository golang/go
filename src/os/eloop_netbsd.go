// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netbsd

package os

import "syscall"

// isNoFollowErr reports whether err may result from O_NOFOLLOW blocking an open operation.
func isNoFollowErr(err error) bool {
	// NetBSD returns EFTYPE, but check the other possibilities as well.
	switch err {
	case syscall.ELOOP, syscall.EMLINK, syscall.EFTYPE:
		return true
	}
	return false
}
