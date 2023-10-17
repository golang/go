// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package execenv

import (
	"internal/syscall/windows"
	"syscall"
	"unsafe"
)

// Default will return the default environment
// variables based on the process attributes
// provided.
//
// If the process attributes contain a token, then
// the environment variables will be sourced from
// the defaults for that user token, otherwise they
// will be sourced from syscall.Environ().
func Default(sys *syscall.SysProcAttr) (env []string, err error) {
	if sys == nil || sys.Token == 0 {
		return syscall.Environ(), nil
	}
	var blockp *uint16
	err = windows.CreateEnvironmentBlock(&blockp, sys.Token, false)
	if err != nil {
		return nil, err
	}
	defer windows.DestroyEnvironmentBlock(blockp)

	const size = unsafe.Sizeof(*blockp)
	for *blockp != 0 { // environment block ends with empty string
		// find NUL terminator
		end := unsafe.Add(unsafe.Pointer(blockp), size)
		for *(*uint16)(end) != 0 {
			end = unsafe.Add(end, size)
		}

		entry := unsafe.Slice(blockp, (uintptr(end)-uintptr(unsafe.Pointer(blockp)))/2)
		env = append(env, syscall.UTF16ToString(entry))
		blockp = (*uint16)(unsafe.Add(end, size))
	}
	return
}
