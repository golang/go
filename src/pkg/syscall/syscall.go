// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package contains an interface to the low-level operating system
// primitives.  The details vary depending on the underlying system.
// Its primary use is inside other packages that provide a more portable
// interface to the system, such as "os", "time" and "net".  Use those
// packages rather than this one if you can.
// For details of the functions and data types in this package consult
// the manuals for the appropriate operating system.
// These calls return errno == 0 to indicate success; otherwise
// errno is an operating system error number describing the failure.
package syscall

import (
	"sync"
	"unsafe"
)

// StringByteSlice returns a NUL-terminated slice of bytes
// containing the text of s.
func StringByteSlice(s string) []byte {
	a := make([]byte, len(s)+1)
	copy(a, s)
	return a
}

// StringBytePtr returns a pointer to a NUL-terminated array of bytes
// containing the text of s.
func StringBytePtr(s string) *byte { return &StringByteSlice(s)[0] }

// Single-word zero for use when we need a valid pointer to 0 bytes.
// See mksyscall.sh.
var _zero uintptr

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
	m.active[p] = nil, false
	return 0
}
