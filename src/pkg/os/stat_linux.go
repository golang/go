// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import "syscall"

func isSymlink(stat *syscall.Stat_t) bool {
	return stat.Mode&syscall.S_IFMT == syscall.S_IFLNK
}

func fileInfoFromStat(name string, fi *FileInfo, lstat, stat *syscall.Stat_t) *FileInfo {
	fi.Dev = stat.Dev
	fi.Ino = uint64(stat.Ino)
	fi.Nlink = uint64(stat.Nlink)
	fi.Mode = stat.Mode
	fi.Uid = stat.Uid
	fi.Gid = stat.Gid
	fi.Rdev = stat.Rdev
	fi.Size = uint64(stat.Size)
	fi.Blksize = uint64(stat.Blksize)
	fi.Blocks = uint64(stat.Blocks)
	fi.Atime_ns = uint64(syscall.TimespecToNsec(stat.Atim))
	fi.Mtime_ns = uint64(syscall.TimespecToNsec(stat.Mtim))
	fi.Ctime_ns = uint64(syscall.TimespecToNsec(stat.Ctim))
	for i := len(name) - 1; i >= 0; i-- {
		if name[i] == '/' {
			name = name[i+1:]
			break
		}
	}
	fi.Name = name
	if isSymlink(lstat) && !isSymlink(stat) {
		fi.FollowedSymlink = true
	}
	return fi
}
