// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (amd64 || arm64)

package runtime_test

import (
	"encoding/binary"
	"syscall"
	"unsafe"
)

const (
	perfTypeHWCache   = 3
	perfL1DReadMiss   = 0 | 0<<8 | 1<<16
	perfAttrSize      = 128
	perfFlagDisabled  = 1 << 0
	perfFlagNoKernel  = 1 << 5
	perfFlagNoHyper   = 1 << 6
	perfFlagCloexec   = 8
	perfIocEnable     = 0x2400
	perfIocDisable    = 0x2401
	perfIocReset      = 0x2403
	perfReadBytesSize = 8
)

// l1Counter counts user-mode L1 data cache read misses of the calling thread.
type l1Counter struct{ fd int }

// openL1Misses opens the counter, or reports why the machine cannot count.
func openL1Misses() (*l1Counter, error) {
	var attr [perfAttrSize / 8]uint64
	attr[0] = perfTypeHWCache | perfAttrSize<<32
	attr[1] = perfL1DReadMiss
	attr[5] = perfFlagDisabled | perfFlagNoKernel | perfFlagNoHyper
	fd, _, errno := syscall.Syscall6(syscall.SYS_PERF_EVENT_OPEN, uintptr(unsafe.Pointer(&attr)), 0, ^uintptr(0), ^uintptr(0), perfFlagCloexec, 0)
	if errno != 0 {
		return nil, errno
	}
	return &l1Counter{fd: int(fd)}, nil
}

func (c *l1Counter) ioctl(req uintptr) {
	syscall.Syscall(syscall.SYS_IOCTL, uintptr(c.fd), req, 0)
}

// start zeroes the count and begins counting.
func (c *l1Counter) start() {
	c.ioctl(perfIocReset)
	c.ioctl(perfIocEnable)
}

// stop ends counting and returns the misses since start, or false when the
// count cannot be read.
func (c *l1Counter) stop() (uint64, bool) {
	c.ioctl(perfIocDisable)
	var buf [perfReadBytesSize]byte
	if n, err := syscall.Read(c.fd, buf[:]); err != nil || n != len(buf) {
		return 0, false
	}
	return binary.NativeEndian.Uint64(buf[:]), true
}

func (c *l1Counter) close() { syscall.Close(c.fd) }
