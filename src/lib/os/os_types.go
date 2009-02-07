// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

// An operating-system independent representation of Unix data structures.
// OS-specific routines in this directory convert the OS-local versions to these.

// Result of stat64(2) etc.
type Dir struct {
	Dev	uint64;
	Ino	uint64;
	Nlink	uint64;
	Mode	uint32;
	Uid	uint32;
	Gid	uint32;
	Rdev	uint64;
	Size	uint64;
	Blksize	uint64;
	Blocks	uint64;
	Atime_ns	uint64;	// nanoseconds since 1970
	Mtime_ns	uint64;	// nanoseconds since 1970
	Ctime_ns	uint64;	// nanoseconds since 1970
	Name	string;
}
