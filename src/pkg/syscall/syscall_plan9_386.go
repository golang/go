// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

func Getpagesize() int { return 0x1000 }

func nanotime() (nsec int64, err error) {
	// TODO(paulzhol):
	// avoid reopening a file descriptor for /dev/bintime on each call,
	// use lower-level calls to avoid allocation.

	var b [8]byte
	nsec = -1

	fd, err := Open("/dev/bintime", O_RDONLY)
	if err != nil {
		return
	}
	defer Close(fd)

	if _, err = Pread(fd, b[:], 0); err != nil {
		return
	}

	if nsec, err = DecodeBintime(b[:]); err != nil {
		return -1, err
	}

	return
}
