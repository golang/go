// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd netbsd openbsd

package syscall

func forkExecPipe(p []int) error {
	return Pipe2(p, O_CLOEXEC)
}
