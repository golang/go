// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import "syscall"

func isConnError(err error) bool {
	if se, ok := err.(syscall.Errno); ok {
		return se == syscall.WSAECONNRESET || se == syscall.WSAECONNABORTED
	}
	return false
}
