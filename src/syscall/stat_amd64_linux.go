// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

func stat(*byte, *Stat) (ret int64, errno int64);
func fstat(int64, *Stat) (ret int64, errno int64);

export Stat
export stat, fstat

// Stat and relatives for Linux

type dev_t uint64;
type ino_t uint64;
type mode_t uint32;
type nlink_t uint64;
type uid_t uint32;
type gid_t uint32;
type off_t int64;
type blksize_t int64;
type blkcnt_t int64;
type time_t int64;

type Timespec struct {
	tv_sec	time_t;
	tv_nsec	int64;
}

type Stat struct {
	st_dev	dev_t;     /* ID of device containing file */
	st_ino	ino_t;     /* inode number */
	st_nlink	nlink_t;   /* number of hard links */
	st_mode	mode_t;    /* protection */
	st_uid	uid_t;     /* user ID of owner */
	st_gid	gid_t;     /* group ID of owner */
	pad0	int32;
	st_rdev	dev_t;    /* device ID (if special file) */
	st_size	off_t;    /* total size, in bytes */
	st_blksize	blksize_t; /* blocksize for filesystem I/O */
	st_blocks	blkcnt_t;  /* number of blocks allocated */
	st_atime	Timespec;   /* time of last access */
	st_mtime	Timespec;   /* time of last modification */
	st_ctime	Timespec;   /* time of last status change */
}

