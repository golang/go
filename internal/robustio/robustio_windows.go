// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package robustio

import (
	"errors"
	"syscall"
)

const errFileNotFound = syscall.ERROR_FILE_NOT_FOUND

// isEphemeralError returns true if err may be resolved by waiting.
func isEphemeralError(err error) bool {
	var errno syscall.Errno
	if errors.As(err, &errno) {
		switch errno {
		case syscall.ERROR_ACCESS_DENIED,
			syscall.ERROR_FILE_NOT_FOUND,
			ERROR_SHARING_VIOLATION:
			return true
		}
	}
	return false
}

func getFileID(filename string) (FileID, error) {
	filename16, err := syscall.UTF16PtrFromString(filename)
	if err != nil {
		return FileID{}, err
	}
	h, err := syscall.CreateFile(filename16, 0, 0, nil, syscall.OPEN_EXISTING, uint32(syscall.FILE_FLAG_BACKUP_SEMANTICS), 0)
	if err != nil {
		return FileID{}, err
	}
	defer syscall.CloseHandle(h)
	var i syscall.ByHandleFileInformation
	if err := syscall.GetFileInformationByHandle(h, &i); err != nil {
		return FileID{}, err
	}
	return FileID{
		device: uint64(i.VolumeSerialNumber),
		inode:  uint64(i.FileIndexHigh)<<32 | uint64(i.FileIndexLow),
	}, nil
}
