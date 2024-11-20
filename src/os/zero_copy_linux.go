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
		written, err, handled = poll.SendFile(pfd, int(fd), 0)
		return true
	})

	if err == nil {
		err = rerr
	}

	return written, handled, wrapSyscallError("sendfile", err)
}

func (f *File) readFrom(r io.Reader) (written int64, handled bool, err error) {
	// Neither copy_file_range(2)/sendfile(2) nor splice(2) supports destinations opened with
	// O_APPEND, so don't bother to try zero-copy with these system calls.
	//
	// Visit https://man7.org/linux/man-pages/man2/copy_file_range.2.html#ERRORS and
	// https://man7.org/linux/man-pages/man2/sendfile.2.html#ERRORS and
	// https://man7.org/linux/man-pages/man2/splice.2.html#ERRORS for details.
	if f.appendMode {
		return 0, false, nil
	}

	written, handled, err = f.copyFile(r)
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

	written, handled, err = pollSplice(&f.pfd, pfd, remain)

	if lr != nil {
		lr.N = remain - written
	}

	return written, handled, wrapSyscallError("splice", err)
}

func (f *File) copyFile(r io.Reader) (written int64, handled bool, err error) {
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

	if handled {
		return written, handled, wrapSyscallError("copy_file_range", err)
	}

	// If fd_in and fd_out refer to the same file and the source and target ranges overlap,
	// copy_file_range(2) just returns EINVAL error. poll.CopyFileRange will ignore that
	// error and act like it didn't call copy_file_range(2). Then the caller will fall back
	// to generic copy, which results in doubling the content in the file.
	// By contrast, sendfile(2) allows this kind of overlapping and works like a memmove,
	// in this case the file content will remain the same after copying, which is not what we want.
	// Thus, we just bail out here and leave it to generic copy when it's a file copying itself.
	if f.pfd.Sysfd == src.pfd.Sysfd {
		return 0, false, nil
	}

	sc, err := src.SyscallConn()
	if err != nil {
		return
	}

	// We can employ sendfile(2) when copy_file_range(2) fails to handle the copy.
	// sendfile(2) enabled file-to-file copying since Linux 2.6.33 and Go requires
	// Linux 3.2 or later, so we're good to go.
	// Check out https://man7.org/linux/man-pages/man2/sendfile.2.html#DESCRIPTION for more details.
	rerr := sc.Read(func(fd uintptr) bool {
		written, err, handled = poll.SendFile(&f.pfd, int(fd), remain)
		return true
	})
	if lr != nil {
		lr.N -= written
	}

	if err == nil {
		err = rerr
	}

	return written, handled, wrapSyscallError("sendfile", err)
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
