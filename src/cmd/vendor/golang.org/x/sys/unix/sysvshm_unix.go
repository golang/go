// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (darwin && !ios) || linux

package unix

import "unsafe"

// SysvShmAttach attaches the Sysv shared memory segment associated with the
// shared memory identifier id.
func SysvShmAttach(id int, addr uintptr, flag int) ([]byte, error) {
	addr, errno := shmat(id, addr, flag)
	if errno != nil {
		return nil, errno
	}

	// Retrieve the size of the shared memory to enable slice creation
	var info SysvShmDesc

	_, err := SysvShmCtl(id, IPC_STAT, &info)
	if err != nil {
		// release the shared memory if we can't find the size

		// ignoring error from shmdt as there's nothing sensible to return here
		shmdt(addr)
		return nil, err
	}

	// Use unsafe to convert addr into a []byte.
	b := unsafe.Slice((*byte)(unsafe.Pointer(addr)), int(info.Segsz))
	return b, nil
}

// SysvShmDetach unmaps the shared memory slice returned from SysvShmAttach.
//
// It is not safe to use the slice after calling this function.
func SysvShmDetach(data []byte) error {
	if len(data) == 0 {
		return EINVAL
	}

	return shmdt(uintptr(unsafe.Pointer(&data[0])))
}

// SysvShmGet returns the Sysv shared memory identifier associated with key.
// If the IPC_CREAT flag is specified a new segment is created.
func SysvShmGet(key, size, flag int) (id int, err error) {
	return shmget(key, size, flag)
}
