// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const (
	_AT_NULL   = 0
	_AT_RANDOM = 25 // introduced in 2.6.29
)

var randomNumber uint32

//go:nosplit
func cputicks() int64 {
	// Currently cputicks() is used in blocking profiler and to seed fastrand1().
	// nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	// randomNumber provides better seeding of fastrand1.
	return nanotime() + int64(randomNumber)
}
