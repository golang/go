// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 *  System structs for Darwin, amd64
 */

typedef uint32 dev_t;
typedef uint64 ino_t;
typedef uint16 mode_t;
typedef uint16 nlink_t;
typedef uint32 uid_t;
typedef uint32 gid_t;
typedef int64 off_t;
typedef int32 blksize_t;
typedef int64 blkcnt_t;
typedef int64 time_t;

struct timespec {
	time_t tv_sec;
	int64 tv_nsec;
};

struct stat {	// really a stat64
	dev_t st_dev;
	mode_t st_mode;
	nlink_t st_nlink;
	ino_t st_ino;
	uid_t st_uid;
	gid_t st_gid;
	dev_t st_rdev;
	struct timespec st_atimespec;
	struct timespec st_mtimespec;
	struct timespec st_ctimespec;
	struct timespec st_birthtimespec;
	off_t st_size;
	blkcnt_t st_blocks;
	blksize_t st_blksize;
	uint32 st_flags;
	uint32 st_gen;
 	int64 st_qspare[2];
};

#define	O_CREAT	0x0200
