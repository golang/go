// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

const (
	// UTIME_OMIT is the sentinel value to indicate that a time value should not
	// be changed. It is useful for example to indicate for example with UtimesNano
	// to avoid changing AccessTime or ModifiedTime.
	// Its value must match syscall/fs_js.go
	UTIME_OMIT = -0x2
)
