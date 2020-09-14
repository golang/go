// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd

package poll

import "syscall"

func newIovecWithBase(base *byte) syscall.Iovec {
	return syscall.Iovec{Base: base}
}
