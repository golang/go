// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import "syscall"

func init() {
	useSyscallwd = useSyscallwdDarwin
}

func useSyscallwdDarwin(err error) bool {
	return err != syscall.ENOTSUP
}
