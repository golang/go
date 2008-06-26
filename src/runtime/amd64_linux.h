// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 *  System structs for Darwin, amd64
 */

typedef uint64 dev_t;
typedef uint64 ino_t;
typedef uint32 mode_t;
typedef uint64 nlink_t;
typedef uint32 uid_t;
typedef uint32 gid_t;
typedef int64 off_t;
typedef int64 blksize_t;
typedef int64 blkcnt_t;
typedef int64 time_t;

struct timespec {
	time_t tv_sec;
	int64 tv_nsec;
};

struct stat {
	dev_t	st_dev;     /* ID of device containing file */
	ino_t	st_ino;     /* inode number */
	mode_t	st_mode;    /* protection */
	nlink_t	st_nlink;   /* number of hard links */
	uid_t	st_uid;     /* user ID of owner */
	gid_t	st_gid;     /* group ID of owner */
	dev_t	st_rdev;
	dev_t	st_rdev;    /* device ID (if special file) */
	struct timespec	st_atime;   /* time of last access */
	struct timespec	st_mtime;   /* time of last modification */
	struct timespec	st_ctime;   /* time of last status change */
	off_t	st_size;    /* total size, in bytes */
	blkcnt_t	st_blocks;  /* number of blocks allocated */
	blksize_t st_blksize; /* blocksize for filesystem I/O */
};
