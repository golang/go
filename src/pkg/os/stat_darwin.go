// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import "syscall"

func isSymlink(stat *syscall.Stat_t) bool {
	return stat.Mode & syscall.S_IFMT == syscall.S_IFLNK
}

func dirFromStat(name string, dir *Dir, lstat, stat *syscall.Stat_t) *Dir {
	dir.Dev = uint64(stat.Dev);
	dir.Ino = stat.Ino;
	dir.Nlink = uint64(stat.Nlink);
	dir.Mode = uint32(stat.Mode);
	dir.Uid = stat.Uid;
	dir.Gid = stat.Gid;
	dir.Rdev = uint64(stat.Rdev);
	dir.Size = uint64(stat.Size);
	dir.Blksize = uint64(stat.Blksize);
	dir.Blocks = uint64(stat.Blocks);
	dir.Atime_ns = uint64(syscall.TimespecToNsec(stat.Atimespec));
	dir.Mtime_ns = uint64(syscall.TimespecToNsec(stat.Mtimespec));
	dir.Ctime_ns = uint64(syscall.TimespecToNsec(stat.Ctimespec));
	for i := len(name)-1; i >= 0; i-- {
		if name[i] == '/' {
			name = name[i+1 : len(name)];
			break;
		}
	}
	dir.Name = name;
	if isSymlink(lstat) && !isSymlink(stat) {
		dir.FollowedSymlink = true
	}
	return dir;
}
