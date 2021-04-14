// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !faketime

package syscall

const faketime = false

func faketimeWrite(fd int, p []byte) int {
	// This should never be called since faketime is false.
	panic("not implemented")
}
