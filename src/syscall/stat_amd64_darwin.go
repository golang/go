// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

func stat(name *byte, buf *Stat) (ret int64, errno int64);
func fstat(fd int64, buf *Stat) (ret int64, errno int64);
func lstat(name *byte, buf *Stat) (ret int64, errno int64);

export Stat
export stat, fstat, lstat

// Stat and relatives for Darwin

type dev_t uint32;
type ino_t uint64;
type mode_t uint16;
type nlink_t uint16;
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
	st_mode	mode_t;    /* protection */
	st_nlink	nlink_t;   /* number of hard links */
	st_ino	ino_t;     /* inode number */
	st_uid	uid_t;     /* user ID of owner */
	st_gid	gid_t;     /* group ID of owner */
	st_rdev	dev_t;    /* device ID (if special file) */
	st_atime	Timespec;   /* time of last access */
	st_mtime	Timespec;   /* time of last modification */
	st_ctime	Timespec;   /* time of last status change */
	st_birthtimespec	Timespec;   /* birth time */
	st_size	off_t;    /* total size, in bytes */
	st_blocks	blkcnt_t;  /* number of blocks allocated */
	st_blksize	blksize_t; /* blocksize for filesystem I/O */
	st_flags	uint32;
	st_gen		uint32;
 	st_qspare[2]	int64;
}
