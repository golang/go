// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || (solaris && !illumos)

// This code implements the filelock API using POSIX 'fcntl' locks, which attach
// to an (inode, process) pair rather than a file descriptor. To avoid unlocking
// files prematurely when the same file is opened through different descriptors,
// we allow only one read-lock at a time.
//
// Most platforms provide some alternative API, such as an 'flock' system call
// or an F_OFD_SETLK command for 'fcntl', that allows for better concurrency and
// does not require per-inode bookkeeping in the application.

package filelock

import (
	"errors"
	"io"
	"io/fs"
	"math/rand"
	"sync"
	"syscall"
	"time"
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
		return &fs.PathError{
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

	// Spurious EDEADLK errors arise on platforms that compute deadlock graphs at
	// the process, rather than thread, level. Consider processes P and Q, with
	// threads P.1, P.2, and Q.3. The following trace is NOT a deadlock, but will be
	// reported as a deadlock on systems that consider only process granularity:
	//
	// 	P.1 locks file A.
	// 	Q.3 locks file B.
	// 	Q.3 blocks on file A.
	// 	P.2 blocks on file B. (This is erroneously reported as a deadlock.)
	// 	P.1 unlocks file A.
	// 	Q.3 unblocks and locks file A.
	// 	Q.3 unlocks files A and B.
	// 	P.2 unblocks and locks file B.
	// 	P.2 unlocks file B.
	//
	// These spurious errors were observed in practice on AIX and Solaris in
	// cmd/go: see https://golang.org/issue/32817.
	//
	// We work around this bug by treating EDEADLK as always spurious. If there
	// really is a lock-ordering bug between the interacting processes, it will
	// become a livelock instead, but that's not appreciably worse than if we had
	// a proper flock implementation (which generally does not even attempt to
	// diagnose deadlocks).
	//
	// In the above example, that changes the trace to:
	//
	// 	P.1 locks file A.
	// 	Q.3 locks file B.
	// 	Q.3 blocks on file A.
	// 	P.2 spuriously fails to lock file B and goes to sleep.
	// 	P.1 unlocks file A.
	// 	Q.3 unblocks and locks file A.
	// 	Q.3 unlocks files A and B.
	// 	P.2 wakes up and locks file B.
	// 	P.2 unlocks file B.
	//
	// We know that the retry loop will not introduce a *spurious* livelock
	// because, according to the POSIX specification, EDEADLK is only to be
	// returned when “the lock is blocked by a lock from another process”.
	// If that process is blocked on some lock that we are holding, then the
	// resulting livelock is due to a real deadlock (and would manifest as such
	// when using, for example, the flock implementation of this package).
	// If the other process is *not* blocked on some other lock that we are
	// holding, then it will eventually release the requested lock.

	nextSleep := 1 * time.Millisecond
	const maxSleep = 500 * time.Millisecond
	for {
		err = setlkw(f.Fd(), lt)
		if err != syscall.EDEADLK {
			break
		}
		time.Sleep(nextSleep)

		nextSleep += nextSleep
		if nextSleep > maxSleep {
			nextSleep = maxSleep
		}
		// Apply 10% jitter to avoid synchronizing collisions when we finally unblock.
		nextSleep += time.Duration((0.1*rand.Float64() - 0.05) * float64(nextSleep))
	}

	if err != nil {
		unlock(f)
		return &fs.PathError{
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
