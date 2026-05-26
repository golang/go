// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || linux || openbsd

package unix

import "unsafe"

// minIovec is the size of the small initial allocation used by
// Readv, Writev, etc.
//
// This small allocation gets stack allocated, which lets the
// common use case of len(iovs) <= minIovec avoid more expensive
// heap allocations.
const minIovec = 8

// appendBytes converts bs to Iovecs and appends them to vecs.
func appendBytes(vecs []Iovec, bs [][]byte) []Iovec {
	for _, b := range bs {
		var v Iovec
		v.SetLen(len(b))
		if len(b) > 0 {
			v.Base = &b[0]
		} else {
			v.Base = (*byte)(unsafe.Pointer(&_zero))
		}
		vecs = append(vecs, v)
	}
	return vecs
}

// writevRaceDetect tells the race detector that the program
// has read the first n bytes stored in iovecs.
func writevRaceDetect(iovecs []Iovec, n int) {
	if !raceenabled {
		return
	}
	for i := 0; n > 0 && i < len(iovecs); i++ {
		m := min(int(iovecs[i].Len), n)
		n -= m
		if m > 0 {
			raceReadRange(unsafe.Pointer(iovecs[i].Base), m)
		}
	}
}

// readvRaceDetect tells the race detector that the program
// has written to the first n bytes stored in iovecs.
func readvRaceDetect(iovecs []Iovec, n int, err error) {
	if !raceenabled {
		return
	}
	for i := 0; n > 0 && i < len(iovecs); i++ {
		m := min(int(iovecs[i].Len), n)
		n -= m
		if m > 0 {
			raceWriteRange(unsafe.Pointer(iovecs[i].Base), m)
		}
	}
	if err == nil {
		raceAcquire(unsafe.Pointer(&ioSync))
	}
}

func Readv(fd int, iovs [][]byte) (n int, err error) {
	iovecs := make([]Iovec, 0, minIovec)
	iovecs = appendBytes(iovecs, iovs)
	n, err = readv(fd, iovecs)
	readvRaceDetect(iovecs, n, err)
	return n, err
}

func Preadv(fd int, iovs [][]byte, offset int64) (n int, err error) {
	iovecs := make([]Iovec, 0, minIovec)
	iovecs = appendBytes(iovecs, iovs)
	n, err = preadv(fd, iovecs, offset)
	readvRaceDetect(iovecs, n, err)
	return n, err
}

func Writev(fd int, iovs [][]byte) (n int, err error) {
	iovecs := make([]Iovec, 0, minIovec)
	iovecs = appendBytes(iovecs, iovs)
	if raceenabled {
		raceReleaseMerge(unsafe.Pointer(&ioSync))
	}
	n, err = writev(fd, iovecs)
	writevRaceDetect(iovecs, n)
	return n, err
}

func Pwritev(fd int, iovs [][]byte, offset int64) (n int, err error) {
	iovecs := make([]Iovec, 0, minIovec)
	iovecs = appendBytes(iovecs, iovs)
	if raceenabled {
		raceReleaseMerge(unsafe.Pointer(&ioSync))
	}
	n, err = pwritev(fd, iovecs, offset)
	writevRaceDetect(iovecs, n)
	return n, err
}
