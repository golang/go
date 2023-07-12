// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux
// +build linux

package unix

import "unsafe"

type mremapMmapper struct {
	mmapper
	mremap func(oldaddr uintptr, oldlength uintptr, newlength uintptr, flags int, newaddr uintptr) (xaddr uintptr, err error)
}

func (m *mremapMmapper) Mremap(oldData []byte, newLength int, flags int) (data []byte, err error) {
	if newLength <= 0 || len(oldData) == 0 || len(oldData) != cap(oldData) || flags&MREMAP_FIXED != 0 {
		return nil, EINVAL
	}

	pOld := &oldData[cap(oldData)-1]
	m.Lock()
	defer m.Unlock()
	bOld := m.active[pOld]
	if bOld == nil || &bOld[0] != &oldData[0] {
		return nil, EINVAL
	}
	newAddr, errno := m.mremap(uintptr(unsafe.Pointer(&bOld[0])), uintptr(len(bOld)), uintptr(newLength), flags, 0)
	if errno != nil {
		return nil, errno
	}
	bNew := unsafe.Slice((*byte)(unsafe.Pointer(newAddr)), newLength)
	pNew := &bNew[cap(bNew)-1]
	if flags&MREMAP_DONTUNMAP == 0 {
		delete(m.active, pOld)
	}
	m.active[pNew] = bNew
	return bNew, nil
}
