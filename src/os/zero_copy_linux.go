// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"internal/poll"
	"io"
	"syscall"
)

var (
	pollCopyFileRange = poll.CopyFileRange
	pollSplice        = poll.Splice
)

func (f *File) writeTo(w io.Writer) (written int64, handled bool, err error) {
	pfd, network := getPollFDAndNetwork(w)
	// TODO(panjf2000): same as File.spliceToFile.
	if pfd == nil || !pfd.IsStream || !isUnixOrTCP(string(network)) {
		return
	}

	sc, err := f.SyscallConn()
	if err != nil {
		return
	}

	rerr := sc.Read(func(fd uintptr) (done bool) {
		written, err, handled = poll.SendFile(pfd, fd, 0)
		return true
	})

	if err == nil {
		err = rerr
	}

	return written, handled, wrapSyscallError("sendfile", err)
}

func (f *File) readFrom(r io.Reader) (written int64, handled bool, err error) {
	// Neither copy_file_range(2) nor splice(2) supports destinations opened with
	// O_APPEND, so don't bother to try zero-copy with these system calls.
	//
	// Visit https://man7.org/linux/man-pages/man2/copy_file_range.2.html#ERRORS and
	// https://man7.org/linux/man-pages/man2/splice.2.html#ERRORS for details.
	if f.appendMode {
		return 0, false, nil
	}

	written, handled, err = f.copyFileRange(r)
	if handled {
		return
	}
	return f.spliceToFile(r)
}

func (f *File) spliceToFile(r io.Reader) (written int64, handled bool, err error) {
	var (
		remain int64
		lr     *io.LimitedReader
	)
	if lr, r, remain = tryLimitedReader(r); remain <= 0 {
		return 0, true, nil
	}

	pfd, _ := getPollFDAndNetwork(r)
	// TODO(panjf2000): run some tests to see if we should unlock the non-streams for splice.
	// Streams benefit the most from the splice(2), non-streams are not even supported in old kernels
	// where splice(2) will just return EINVAL; newer kernels support non-streams like UDP, but I really
	// doubt that splice(2) could help non-streams, cuz they usually send small frames respectively
	// and one splice call would result in one frame.
	// splice(2) is suitable for large data but the generation of fragments defeats its edge here.
	// Therefore, don't bother to try splice if the r is not a streaming descriptor.
	if pfd == nil || !pfd.IsStream {
		return
	}

	// Don't use splice to a pipe, since it can lead to a busy loop if the
	// reader is not reading.
	// See issue 68303.
	fi, err := f.Stat()
	if err == nil && fi.Mode()&ModeNamedPipe != 0 {
		return 0, false, nil
	}

	written, handled, err = pollSplice(&f.pfd, pfd, remain)

	if lr != nil {
		lr.N = remain - written
	}

	return written, handled, wrapSyscallError("splice", err)
}

func (f *File) copyFileRange(r io.Reader) (written int64, handled bool, err error) {
	var (
		remain int64
		lr     *io.LimitedReader
	)
	if lr, r, remain = tryLimitedReader(r); remain <= 0 {
		return 0, true, nil
	}

	var src *File
	switch v := r.(type) {
	case *File:
		src = v
	case fileWithoutWriteTo:
		src = v.File
	default:
		return 0, false, nil
	}

	if src.checkValid("ReadFrom") != nil {
		// Avoid returning the error as we report handled as false,
		// leave further error handling as the responsibility of the caller.
		return 0, false, nil
	}

	written, handled, err = pollCopyFileRange(&f.pfd, &src.pfd, remain)
	if lr != nil {
		lr.N -= written
	}
	return written, handled, wrapSyscallError("copy_file_range", err)
}

// getPollFDAndNetwork tries to get the poll.FD and network type from the given interface
// by expecting the underlying type of i to be the implementation of syscall.Conn
// that contains a *net.rawConn.
func getPollFDAndNetwork(i any) (*poll.FD, poll.String) {
	sc, ok := i.(syscall.Conn)
	if !ok {
		return nil, ""
	}
	rc, err := sc.SyscallConn()
	if err != nil {
		return nil, ""
	}
	irc, ok := rc.(interface {
		PollFD() *poll.FD
		Network() poll.String
	})
	if !ok {
		return nil, ""
	}
	return irc.PollFD(), irc.Network()
}

func isUnixOrTCP(network string) bool {
	switch network {
	case "tcp", "tcp4", "tcp6", "unix":
		return true
	default:
		return false
	}
}
