// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix solaris

// This code implements the filelock API using POSIX 'fcntl' locks, which attach
// to an (inode, process) pair rather than a file descriptor. To avoid unlocking
// files prematurely when the same file is opened through different descriptors,
// we allow only one read-lock at a time.
//
// Most platforms provide some alternative API, such as an 'flock' system call
// or an F_OFD_SETLK command for 'fcntl', that allows for better concurrency and
// does not require per-inode bookkeeping in the application.
//
// TODO(bcmills): If we add a build tag for Illumos (see golang.org/issue/20603)
// then Illumos should use F_OFD_SETLK, and the resulting code would be as
// simple as filelock_unix.go. We will still need the code in this file for AIX
// or as long as Oracle Solaris provides only F_SETLK.

package filelock

import (
	"errors"
	"io"
	"os"
	"sync"
	"syscall"
)

type lockType int16

const (
	readLock  lockType = syscall.F_RDLCK
	writeLock lockType = syscall.F_WRLCK
)

type inode = uint64 // type of syscall.Stat_t.Ino

type inodeLock struct {
	owner File
	queue []<-chan File
}

type token struct{}

var (
	mu     sync.Mutex
	inodes = map[File]inode{}
	locks  = map[inode]inodeLock{}
)

func lock(f File, lt lockType) (err error) {
	// POSIX locks apply per inode and process, and the lock for an inode is
	// released when *any* descriptor for that inode is closed. So we need to
	// synchronize access to each inode internally, and must serialize lock and
	// unlock calls that refer to the same inode through different descriptors.
	fi, err := f.Stat()
	if err != nil {
		return err
	}
	ino := fi.Sys().(*syscall.Stat_t).Ino

	mu.Lock()
	if i, dup := inodes[f]; dup && i != ino {
		mu.Unlock()
		return &os.PathError{
			Op:   lt.String(),
			Path: f.Name(),
			Err:  errors.New("inode for file changed since last Lock or RLock"),
		}
	}
	inodes[f] = ino

	var wait chan File
	l := locks[ino]
	if l.owner == f {
		// This file already owns the lock, but the call may change its lock type.
	} else if l.owner == nil {
		// No owner: it's ours now.
		l.owner = f
	} else {
		// Already owned: add a channel to wait on.
		wait = make(chan File)
		l.queue = append(l.queue, wait)
	}
	locks[ino] = l
	mu.Unlock()

	if wait != nil {
		wait <- f
	}

	err = setlkw(f.Fd(), lt)

	if err != nil {
		unlock(f)
		return &os.PathError{
			Op:   lt.String(),
			Path: f.Name(),
			Err:  err,
		}
	}

	return nil
}

func unlock(f File) error {
	var owner File

	mu.Lock()
	ino, ok := inodes[f]
	if ok {
		owner = locks[ino].owner
	}
	mu.Unlock()

	if owner != f {
		panic("unlock called on a file that is not locked")
	}

	err := setlkw(f.Fd(), syscall.F_UNLCK)

	mu.Lock()
	l := locks[ino]
	if len(l.queue) == 0 {
		// No waiters: remove the map entry.
		delete(locks, ino)
	} else {
		// The first waiter is sending us their file now.
		// Receive it and update the queue.
		l.owner = <-l.queue[0]
		l.queue = l.queue[1:]
		locks[ino] = l
	}
	delete(inodes, f)
	mu.Unlock()

	return err
}

// setlkw calls FcntlFlock with F_SETLKW for the entire file indicated by fd.
func setlkw(fd uintptr, lt lockType) error {
	for {
		err := syscall.FcntlFlock(fd, syscall.F_SETLKW, &syscall.Flock_t{
			Type:   int16(lt),
			Whence: io.SeekStart,
			Start:  0,
			Len:    0, // All bytes.
		})
		if err != syscall.EINTR {
			return err
		}
	}
}

func isNotSupported(err error) bool {
	return err == syscall.ENOSYS || err == syscall.ENOTSUP || err == syscall.EOPNOTSUPP || err == ErrNotSupported
}
