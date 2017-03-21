// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

func forkExecPipe(p []int) error {
	err := Pipe2(p, O_CLOEXEC)
	if err == nil {
		return nil
	}

	// FreeBSD 9 fallback.
	// TODO: remove this for Go 1.10 per Issue 19072
	err = Pipe(p)
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
