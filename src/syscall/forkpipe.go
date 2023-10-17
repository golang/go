// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin

package syscall

// forkExecPipe opens a pipe and non-atomically sets O_CLOEXEC on both file
// descriptors.
func forkExecPipe(p []int) error {
	err := Pipe(p)
	if err != nil {
		return err
	}
	_, err = fcntl(p[0], F_SETFD, FD_CLOEXEC)
	if err != nil {
		return err
	}
	_, err = fcntl(p[1], F_SETFD, FD_CLOEXEC)
	return err
}

func acquireForkLock() {
	ForkLock.Lock()
}

func releaseForkLock() {
	ForkLock.Unlock()
}
