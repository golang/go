// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"internal/syscall/windows"
	"syscall"
	"unicode/utf16"
	"unsafe"
)

func environForSysProcAttr(sys *syscall.SysProcAttr) (env []string, err error) {
	if sys == nil || sys.Token == 0 {
		return Environ(), nil
	}
	var block *uint16
	err = windows.CreateEnvironmentBlock(&block, sys.Token, false)
	if err != nil {
		return nil, err
	}
	defer windows.DestroyEnvironmentBlock(block)
	blockp := uintptr(unsafe.Pointer(block))
	for {

		// find NUL terminator
		end := unsafe.Pointer(blockp)
		for *(*uint16)(end) != 0 {
			end = unsafe.Pointer(uintptr(end) + 2)
		}

		n := (uintptr(end) - uintptr(unsafe.Pointer(blockp))) / 2
		if n == 0 {
			// environment block ends with empty string
			break
		}

		entry := (*[(1 << 30) - 1]uint16)(unsafe.Pointer(blockp))[:n:n]
		env = append(env, string(utf16.Decode(entry)))
		blockp += 2 * (uintptr(len(entry)) + 1)
	}
	return
}
