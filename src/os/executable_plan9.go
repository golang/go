// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build plan9

package os

import (
	"internal/itoa"
	"syscall"
)

func executable() (string, error) {
	fn := "/proc/" + itoa.Itoa(Getpid()) + "/text"
	f, err := Open(fn)
	if err != nil {
		return "", err
	}
	defer f.Close()
	return syscall.Fd2path(int(f.Fd()))
}
