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
		entry := (*[(1 << 30) - 1]uint16)(unsafe.Pointer(blockp))[:]
		for i, v := range entry {
			if v == 0 {
				entry = entry[:i]
				break
			}
		}
		if len(entry) == 0 {
			break
		}
		env = append(env, string(utf16.Decode(entry)))
		blockp += 2 * (uintptr(len(entry)) + 1)
	}
	return
}
