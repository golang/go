// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package unix

const (
	R_OK = 0x4
	W_OK = 0x2
	X_OK = 0x1

	// NoFollowErrno is the error returned from open/openat called with
	// O_NOFOLLOW flag, when the trailing component (basename) of the path
	// is a symbolic link.
	NoFollowErrno = noFollowErrno
)
