// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import (
	"syscall"
	"unsafe"
)

// IsNonblock returns whether the file descriptor fd is opened
// in non-blocking mode, that is, the [windows.O_FILE_FLAG_OVERLAPPED] flag
// was set when the file was opened.
func IsNonblock(fd syscall.Handle) (nonblocking bool, err error) {
	var info FILE_MODE_INFORMATION
	if err := NtQueryInformationFile(syscall.Handle(fd), &IO_STATUS_BLOCK{}, unsafe.Pointer(&info), uint32(unsafe.Sizeof(info)), FileModeInformation); err != nil {
		return false, err
	}
	return info.Mode&(FILE_SYNCHRONOUS_IO_ALERT|FILE_SYNCHRONOUS_IO_NONALERT) == 0, nil
}
