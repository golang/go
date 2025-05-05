// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

const (
	SYS_CLOSE         = 6
	SYS_FCNTL         = 55
	SYS_MPROTECT      = 125
	SYS_PRCTL         = 172
	SYS_EPOLL_CTL     = 255
	SYS_EPOLL_PWAIT   = 319
	SYS_EPOLL_CREATE1 = 329
	SYS_EPOLL_PWAIT2  = 441
	SYS_EVENTFD2      = 328
	SYS_OPENAT        = 295
	SYS_PREAD64       = 180
	SYS_READ          = 3

	EFD_NONBLOCK = 0x800

	O_LARGEFILE = 0x8000
)

type EpollEvent struct {
	Events uint32
	Data   [8]byte // to match amd64
}
