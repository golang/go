// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build faketime && !windows

// Faketime isn't currently supported on Windows. This would require
// modifying syscall.Write to call syscall.faketimeWrite,
// translating the Stdout and Stderr handles into FDs 1 and 2.
// (See CL 192739 PS 3.)

package runtime

import "unsafe"

// faketime is the simulated time in nanoseconds since 1970 for the
// playground.
var faketime int64 = 1257894000000000000

var faketimeState struct {
	lock mutex

	// lastfaketime is the last faketime value written to fd 1 or 2.
	lastfaketime int64

	// lastfd is the fd to which lastfaketime was written.
	//
	// Subsequent writes to the same fd may use the same
	// timestamp, but the timestamp must increase if the fd
	// changes.
	lastfd uintptr
}

//go:linkname nanotime
//go:nosplit
func nanotime() int64 {
	return faketime
}

//go:linkname time_now time.now
func time_now() (sec int64, nsec int32, mono int64) {
	return faketime / 1e9, int32(faketime % 1e9), faketime
}

// write is like the Unix write system call.
// We have to avoid write barriers to avoid potential deadlock
// on write calls.
//
//go:nowritebarrierrec
func write(fd uintptr, p unsafe.Pointer, n int32) int32 {
	if !(fd == 1 || fd == 2) {
		// Do an ordinary write.
		return write1(fd, p, n)
	}

	// Write with the playback header.

	// First, lock to avoid interleaving writes.
	lock(&faketimeState.lock)

	// If the current fd doesn't match the fd of the previous write,
	// ensure that the timestamp is strictly greater. That way, we can
	// recover the original order even if we read the fds separately.
	t := faketimeState.lastfaketime
	if fd != faketimeState.lastfd {
		t++
		faketimeState.lastfd = fd
	}
	if faketime > t {
		t = faketime
	}
	faketimeState.lastfaketime = t

	// Playback header: 0 0 P B <8-byte time> <4-byte data length> (big endian)
	var buf [4 + 8 + 4]byte
	buf[2] = 'P'
	buf[3] = 'B'
	tu := uint64(t)
	buf[4] = byte(tu >> (7 * 8))
	buf[5] = byte(tu >> (6 * 8))
	buf[6] = byte(tu >> (5 * 8))
	buf[7] = byte(tu >> (4 * 8))
	buf[8] = byte(tu >> (3 * 8))
	buf[9] = byte(tu >> (2 * 8))
	buf[10] = byte(tu >> (1 * 8))
	buf[11] = byte(tu >> (0 * 8))
	nu := uint32(n)
	buf[12] = byte(nu >> (3 * 8))
	buf[13] = byte(nu >> (2 * 8))
	buf[14] = byte(nu >> (1 * 8))
	buf[15] = byte(nu >> (0 * 8))
	write1(fd, unsafe.Pointer(&buf[0]), int32(len(buf)))

	// Write actual data.
	res := write1(fd, p, n)

	unlock(&faketimeState.lock)
	return res
}
