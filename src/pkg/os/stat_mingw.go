// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import "syscall"

func fileInfoFromStat(name string, fi *FileInfo, lstat, stat *syscall.Stat_t) *FileInfo {
	fi.Mode = 0
	if stat.Windata.FileAttributes == syscall.FILE_ATTRIBUTE_DIRECTORY {
		fi.Mode = fi.Mode | syscall.S_IFDIR
	}
	if stat.Windata.FileAttributes == syscall.FILE_ATTRIBUTE_NORMAL {
		fi.Mode = fi.Mode | syscall.S_IFREG
	}
	if stat.Windata.FileAttributes == syscall.FILE_ATTRIBUTE_READONLY {
		fi.Mode = fi.Mode | 0444
	} else {
		fi.Mode = fi.Mode | 0666
	}
	fi.Size = uint64(stat.Windata.FileSizeHigh)<<32 + uint64(stat.Windata.FileSizeLow)
	fi.Name = string(syscall.UTF16ToString(stat.Windata.FileName[0:]))
	fi.FollowedSymlink = false
	// TODO(brainman): use CreationTime LastAccessTime LastWriteTime to prime following Dir fields
	fi.Atime_ns = 0
	fi.Mtime_ns = 0
	fi.Ctime_ns = 0
	return fi
}
