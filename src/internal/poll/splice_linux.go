// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import (
	"internal/syscall/unix"
	"runtime"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"
)

const (
	// spliceNonblock makes calls to splice(2) non-blocking.
	spliceNonblock = 0x2

	// maxSpliceSize is the maximum amount of data Splice asks
	// the kernel to move in a single call to splice(2).
	maxSpliceSize = 4 << 20
)

// Splice transfers at most remain bytes of data from src to dst, using the
// splice system call to minimize copies of data from and to userspace.
//
// Splice gets a pipe buffer from the pool or creates a new one if needed, to serve as a buffer for the data transfer.
// src and dst must both be stream-oriented sockets.
//
// If err != nil, sc is the system call which caused the error.
func Splice(dst, src *FD, remain int64) (written int64, handled bool, sc string, err error) {
	p, sc, err := getPipe()
	if err != nil {
		return 0, false, sc, err
	}
	defer putPipe(p)
	var inPipe, n int
	for err == nil && remain > 0 {
		max := maxSpliceSize
		if int64(max) > remain {
			max = int(remain)
		}
		inPipe, err = spliceDrain(p.wfd, src, max)
		// The operation is considered handled if splice returns no
		// error, or an error other than EINVAL. An EINVAL means the
		// kernel does not support splice for the socket type of src.
		// The failed syscall does not consume any data so it is safe
		// to fall back to a generic copy.
		//
		// spliceDrain should never return EAGAIN, so if err != nil,
		// Splice cannot continue.
		//
		// If inPipe == 0 && err == nil, src is at EOF, and the
		// transfer is complete.
		handled = handled || (err != syscall.EINVAL)
		if err != nil || inPipe == 0 {
			break
		}
		p.data += inPipe

		n, err = splicePump(dst, p.rfd, inPipe)
		if n > 0 {
			written += int64(n)
			remain -= int64(n)
			p.data -= n
		}
	}
	if err != nil {
		return written, handled, "splice", err
	}
	return written, true, "", nil
}

// spliceDrain moves data from a socket to a pipe.
//
// Invariant: when entering spliceDrain, the pipe is empty. It is either in its
// initial state, or splicePump has emptied it previously.
//
// Given this, spliceDrain can reasonably assume that the pipe is ready for
// writing, so if splice returns EAGAIN, it must be because the socket is not
// ready for reading.
//
// If spliceDrain returns (0, nil), src is at EOF.
func spliceDrain(pipefd int, sock *FD, max int) (int, error) {
	if err := sock.readLock(); err != nil {
		return 0, err
	}
	defer sock.readUnlock()
	if err := sock.pd.prepareRead(sock.isFile); err != nil {
		return 0, err
	}
	for {
		n, err := splice(pipefd, sock.Sysfd, max, spliceNonblock)
		if err == syscall.EINTR {
			continue
		}
		if err != syscall.EAGAIN {
			return n, err
		}
		if err := sock.pd.waitRead(sock.isFile); err != nil {
			return n, err
		}
	}
}

// splicePump moves all the buffered data from a pipe to a socket.
//
// Invariant: when entering splicePump, there are exactly inPipe
// bytes of data in the pipe, from a previous call to spliceDrain.
//
// By analogy to the condition from spliceDrain, splicePump
// only needs to poll the socket for readiness, if splice returns
// EAGAIN.
//
// If splicePump cannot move all the data in a single call to
// splice(2), it loops over the buffered data until it has written
// all of it to the socket. This behavior is similar to the Write
// step of an io.Copy in userspace.
func splicePump(sock *FD, pipefd int, inPipe int) (int, error) {
	if err := sock.writeLock(); err != nil {
		return 0, err
	}
	defer sock.writeUnlock()
	if err := sock.pd.prepareWrite(sock.isFile); err != nil {
		return 0, err
	}
	written := 0
	for inPipe > 0 {
		n, err := splice(sock.Sysfd, pipefd, inPipe, spliceNonblock)
		// Here, the condition n == 0 && err == nil should never be
		// observed, since Splice controls the write side of the pipe.
		if n > 0 {
			inPipe -= n
			written += n
			continue
		}
		if err != syscall.EAGAIN {
			return written, err
		}
		if err := sock.pd.waitWrite(sock.isFile); err != nil {
			return written, err
		}
	}
	return written, nil
}

// splice wraps the splice system call. Since the current implementation
// only uses splice on sockets and pipes, the offset arguments are unused.
// splice returns int instead of int64, because callers never ask it to
// move more data in a single call than can fit in an int32.
func splice(out int, in int, max int, flags int) (int, error) {
	n, err := syscall.Splice(in, nil, out, nil, max, flags)
	return int(n), err
}

type splicePipeFields struct {
	rfd  int
	wfd  int
	data int
}

type splicePipe struct {
	splicePipeFields

	// We want to use a finalizer, so ensure that the size is
	// large enough to not use the tiny allocator.
	_ [24 - unsafe.Sizeof(splicePipeFields{})%24]byte
}

// splicePipePool caches pipes to avoid high-frequency construction and destruction of pipe buffers.
// The garbage collector will free all pipes in the sync.Pool periodically, thus we need to set up
// a finalizer for each pipe to close its file descriptors before the actual GC.
var splicePipePool = sync.Pool{New: newPoolPipe}

func newPoolPipe() any {
	// Discard the error which occurred during the creation of pipe buffer,
	// redirecting the data transmission to the conventional way utilizing read() + write() as a fallback.
	p := newPipe()
	if p == nil {
		return nil
	}
	runtime.SetFinalizer(p, destroyPipe)
	return p
}

// getPipe tries to acquire a pipe buffer from the pool or create a new one with newPipe() if it gets nil from the cache.
//
// Note that it may fail to create a new pipe buffer by newPipe(), in which case getPipe() will return a generic error
// and system call name splice in a string as the indication.
func getPipe() (*splicePipe, string, error) {
	v := splicePipePool.Get()
	if v == nil {
		return nil, "splice", syscall.EINVAL
	}
	return v.(*splicePipe), "", nil
}

func putPipe(p *splicePipe) {
	// If there is still data left in the pipe,
	// then close and discard it instead of putting it back into the pool.
	if p.data != 0 {
		runtime.SetFinalizer(p, nil)
		destroyPipe(p)
		return
	}
	splicePipePool.Put(p)
}

var disableSplice unsafe.Pointer

// newPipe sets up a pipe for a splice operation.
func newPipe() (sp *splicePipe) {
	p := (*bool)(atomic.LoadPointer(&disableSplice))
	if p != nil && *p {
		return nil
	}

	var fds [2]int
	// pipe2 was added in 2.6.27 and our minimum requirement is 2.6.23, so it
	// might not be implemented. Falling back to pipe is possible, but prior to
	// 2.6.29 splice returns -EAGAIN instead of 0 when the connection is
	// closed.
	const flags = syscall.O_CLOEXEC | syscall.O_NONBLOCK
	if err := syscall.Pipe2(fds[:], flags); err != nil {
		return nil
	}

	sp = &splicePipe{splicePipeFields: splicePipeFields{rfd: fds[0], wfd: fds[1]}}

	if p == nil {
		p = new(bool)
		defer atomic.StorePointer(&disableSplice, unsafe.Pointer(p))

		// F_GETPIPE_SZ was added in 2.6.35, which does not have the -EAGAIN bug.
		if _, _, errno := syscall.Syscall(unix.FcntlSyscall, uintptr(fds[0]), syscall.F_GETPIPE_SZ, 0); errno != 0 {
			*p = true
			destroyPipe(sp)
			return nil
		}
	}

	return
}

// destroyPipe destroys a pipe.
func destroyPipe(p *splicePipe) {
	CloseFunc(p.rfd)
	CloseFunc(p.wfd)
}
