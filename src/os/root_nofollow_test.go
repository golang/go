// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || wasip1

package os_test

import "syscall"

func init() {
	openNoFollowFlag = syscall.O_NOFOLLOW
}
