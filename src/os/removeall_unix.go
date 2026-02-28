// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || wasip1

package os

import (
	"internal/syscall/unix"
)

func isErrNoFollow(err error) bool {
	return err == unix.NoFollowErrno
}

func newDirFile(fd int, name string) (*File, error) {
	// We use kindNoPoll because we know that this is a directory.
	return newFile(fd, name, kindNoPoll, false), nil
}
