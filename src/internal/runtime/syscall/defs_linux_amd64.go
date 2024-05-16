// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

const (
	SYS_MPROTECT      = 10
	SYS_FCNTL         = 72
	SYS_EPOLL_CTL     = 233
	SYS_EPOLL_PWAIT   = 281
	SYS_EPOLL_CREATE1 = 291
	SYS_EPOLL_PWAIT2  = 441
	SYS_EVENTFD2      = 290

	EFD_NONBLOCK = 0x800
)

type EpollEvent struct {
	Events uint32
	Data   [8]byte // unaligned uintptr
}
