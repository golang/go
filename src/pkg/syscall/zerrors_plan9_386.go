// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

// Constants
const (
	// Invented values to support what package os expects.
	O_CREAT    = 0x02000
	O_APPEND   = 0x00400
	O_NOCTTY   = 0x00000
	O_NONBLOCK = 0x00000
	O_SYNC     = 0x00000
	O_ASYNC    = 0x00000

	S_IFMT   = 0x1f000
	S_IFIFO  = 0x1000
	S_IFCHR  = 0x2000
	S_IFDIR  = 0x4000
	S_IFBLK  = 0x6000
	S_IFREG  = 0x8000
	S_IFLNK  = 0xa000
	S_IFSOCK = 0xc000
)

// Error table
