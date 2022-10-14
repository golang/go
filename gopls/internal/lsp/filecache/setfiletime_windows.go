// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows
// +build windows

package filecache

import (
	"syscall"
	"time"
)

// setFileTime updates the access and modification times of a file.
func setFileTime(filename string, atime, mtime time.Time) error {
	// Latency of this function was measured on the builder
	// at median=1.9ms 90%=6.8ms 95%=12ms.

	filename16, err := syscall.UTF16PtrFromString(filename)
	if err != nil {
		return err
	}
	// See https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-setfiletime
	h, err := syscall.CreateFile(filename16, syscall.FILE_WRITE_ATTRIBUTES, syscall.FILE_SHARE_WRITE, nil, syscall.OPEN_EXISTING, 0, 0)
	if err != nil {
		return err
	}
	defer syscall.Close(h) // ignore error
	afiletime := syscall.NsecToFiletime(atime.UnixNano())
	mfiletime := syscall.NsecToFiletime(mtime.UnixNano())
	return syscall.SetFileTime(h, nil, &afiletime, &mfiletime)
}
