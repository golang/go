// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

const (
	SYS_FCNTL         = 55
	SYS_MPROTECT      = 125
	SYS_EPOLL_CTL     = 250
	SYS_EPOLL_PWAIT   = 312
	SYS_EPOLL_CREATE1 = 327
	SYS_EPOLL_PWAIT2  = 441
	SYS_EVENTFD2      = 323

	EFD_NONBLOCK = 0x800
)

type EpollEvent struct {
	Events    uint32
	pad_cgo_0 [4]byte
	Data      [8]byte // unaligned uintptr
}
