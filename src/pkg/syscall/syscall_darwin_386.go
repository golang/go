// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

func Getpagesize() int { return 4096 }

func TimespecToNsec(ts Timespec) int64 { return int64(ts.Sec)*1e9 + int64(ts.Nsec) }

func NsecToTimespec(nsec int64) (ts Timespec) {
	ts.Sec = int32(nsec / 1e9)
	ts.Nsec = int32(nsec % 1e9)
	return
}

func TimevalToNsec(tv Timeval) int64 { return int64(tv.Sec)*1e9 + int64(tv.Usec)*1e3 }

func NsecToTimeval(nsec int64) (tv Timeval) {
	nsec += 999 // round up to microsecond
	tv.Usec = int32(nsec % 1e9 / 1e3)
	tv.Sec = int32(nsec / 1e9)
	return
}

//sysnb	gettimeofday(tp *Timeval) (sec int32, usec int32, err error)
func Gettimeofday(tv *Timeval) (err error) {
	// The tv passed to gettimeofday must be non-nil
	// but is otherwise unused.  The answers come back
	// in the two registers.
	sec, usec, err := gettimeofday(tv)
	tv.Sec = int32(sec)
	tv.Usec = int32(usec)
	return err
}

func SetKevent(k *Kevent_t, fd, mode, flags int) {
	k.Ident = uint32(fd)
	k.Filter = int16(mode)
	k.Flags = uint16(flags)
}

func (iov *Iovec) SetLen(length int) {
	iov.Len = uint32(length)
}

func (msghdr *Msghdr) SetControllen(length int) {
	msghdr.Controllen = uint32(length)
}

func (cmsg *Cmsghdr) SetLen(length int) {
	cmsg.Len = uint32(length)
}

func Syscall9(num, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2 uintptr, err Errno) // sic
