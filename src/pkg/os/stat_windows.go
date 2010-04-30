// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import "syscall"

func fileInfoFromStat(name string, fi *FileInfo, lstat, stat *syscall.Stat_t) *FileInfo {
	return fileInfoFromWin32finddata(fi, &stat.Windata)
}

func fileInfoFromWin32finddata(fi *FileInfo, d *syscall.Win32finddata) *FileInfo {
	return setFileInfo(fi, string(syscall.UTF16ToString(d.FileName[0:])), d.FileAttributes, d.FileSizeHigh, d.FileSizeLow, d.CreationTime, d.LastAccessTime, d.LastWriteTime)
}

func fileInfoFromByHandleInfo(fi *FileInfo, name string, d *syscall.ByHandleFileInformation) *FileInfo {
	for i := len(name) - 1; i >= 0; i-- {
		if name[i] == '/' || name[i] == '\\' {
			name = name[i+1:]
			break
		}
	}
	return setFileInfo(fi, name, d.FileAttributes, d.FileSizeHigh, d.FileSizeLow, d.CreationTime, d.LastAccessTime, d.LastWriteTime)
}

func setFileInfo(fi *FileInfo, name string, fa, sizehi, sizelo uint32, ctime, atime, wtime syscall.Filetime) *FileInfo {
	fi.Mode = 0
	if fa == syscall.FILE_ATTRIBUTE_DIRECTORY {
		fi.Mode = fi.Mode | syscall.S_IFDIR
	} else {
		fi.Mode = fi.Mode | syscall.S_IFREG
	}
	if fa == syscall.FILE_ATTRIBUTE_READONLY {
		fi.Mode = fi.Mode | 0444
	} else {
		fi.Mode = fi.Mode | 0666
	}
	fi.Size = int64(sizehi)<<32 + int64(sizelo)
	fi.Name = name
	fi.FollowedSymlink = false
	// TODO(brainman): use ctime atime wtime to prime following FileInfo fields
	fi.Atime_ns = 0
	fi.Mtime_ns = 0
	fi.Ctime_ns = 0
	return fi
}
