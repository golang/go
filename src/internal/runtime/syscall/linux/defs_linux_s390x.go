// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package linux

const (
	SYS_CLOSE         = 6
	SYS_FCNTL         = 55
	SYS_MPROTECT      = 125
	SYS_PRCTL         = 172
	SYS_EPOLL_CTL     = 250
	SYS_EPOLL_PWAIT   = 312
	SYS_EPOLL_CREATE1 = 327
	SYS_EPOLL_PWAIT2  = 441
	SYS_EVENTFD2      = 323
	SYS_OPENAT        = 288
	SYS_PREAD64       = 180
	SYS_READ          = 3

	EFD_NONBLOCK = 0x800

	O_LARGEFILE = 0x0
)

type EpollEvent struct {
	Events    uint32
	pad_cgo_0 [4]byte
	Data      [8]byte // unaligned uintptr
}
