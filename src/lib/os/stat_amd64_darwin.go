// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// AMD64, Darwin

package os

import syscall "syscall"
import os "os"

func dirFromStat(name string, dir *Dir, stat *syscall.Stat_t) *Dir {
	dir.Dev = uint64(stat.Dev);
	dir.Ino = stat.Ino;
	dir.Nlink = uint64(stat.Nlink);
	dir.Mode = uint32(stat.Mode);
	dir.Uid = stat.Uid;
	dir.Gid = stat.Gid;
	dir.Rdev = uint64(stat.Rdev);
	dir.Size = stat.Size;
	dir.Blksize = uint64(stat.Blksize);
	dir.Blocks = stat.Blocks;
	dir.Atime_ns = uint64(stat.Atime.Sec) * 1e9 + stat.Atime.Nsec;
	dir.Mtime_ns = uint64(stat.Mtime.Sec) * 1e9 + stat.Mtime.Nsec;
	dir.Ctime_ns = uint64(stat.Ctime.Sec) * 1e9 + stat.Atime.Nsec;
	for i := len(name) - 1; i >= 0; i-- {
		if name[i] == '/' {
			name = name[i+1:len(name)];
			break;
		}
	}
	dir.Name = name;
	return dir;
}
