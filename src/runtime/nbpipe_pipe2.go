// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || linux || netbsd || openbsd || solaris

package runtime

func nonblockingPipe() (r, w int32, errno int32) {
	return pipe2(_O_NONBLOCK | _O_CLOEXEC)
}
