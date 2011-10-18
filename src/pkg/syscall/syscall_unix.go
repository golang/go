// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux openbsd

package syscall

import (
	"sync"
	"unsafe"
)

var (
	Stdin  = 0
	Stdout = 1
	Stderr = 2
)

const darwinAMD64 = OS == "darwin" && ARCH == "amd64"

func Syscall(trap, a1, a2, a3 uintptr) (r1, r2, err uintptr)
func Syscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)
func RawSyscall(trap, a1, a2, a3 uintptr) (r1, r2, err uintptr)
func RawSyscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)

func Errstr(errno int) string {
	if errno < 0 || errno >= int(len(errors)) {
		return "error " + itoa(errno)
	}
	return errors[errno]
}

// Mmap manager, for use by operating system-specific implementations.

type mmapper struct {
	sync.Mutex
	active map[*byte][]byte // active mappings; key is last byte in mapping
	mmap   func(addr, length uintptr, prot, flags, fd int, offset int64) (uintptr, int)
	munmap func(addr uintptr, length uintptr) int
}

func (m *mmapper) Mmap(fd int, offset int64, length int, prot int, flags int) (data []byte, errno int) {
	if length <= 0 {
		return nil, EINVAL
	}

	// Map the requested memory.
	addr, errno := m.mmap(0, uintptr(length), prot, flags, fd, offset)
	if errno != 0 {
		return nil, errno
	}

	// Slice memory layout
	var sl = struct {
		addr uintptr
		len  int
		cap  int
	}{addr, length, length}

	// Use unsafe to turn sl into a []byte.
	b := *(*[]byte)(unsafe.Pointer(&sl))

	// Register mapping in m and return it.
	p := &b[cap(b)-1]
	m.Lock()
	defer m.Unlock()
	m.active[p] = b
	return b, 0
}

func (m *mmapper) Munmap(data []byte) (errno int) {
	if len(data) == 0 || len(data) != cap(data) {
		return EINVAL
	}

	// Find the base of the mapping.
	p := &data[cap(data)-1]
	m.Lock()
	defer m.Unlock()
	b := m.active[p]
	if b == nil || &b[0] != &data[0] {
		return EINVAL
	}

	// Unmap the memory and update m.
	if errno := m.munmap(uintptr(unsafe.Pointer(&b[0])), uintptr(len(b))); errno != 0 {
		return errno
	}
	delete(m.active, p)
	return 0
}
