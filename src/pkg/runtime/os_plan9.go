// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

func pread(fd int32, buf unsafe.Pointer, nbytes int32, offset int64) int32
func pwrite(fd int32, buf unsafe.Pointer, nbytes int32, offset int64) int32
func seek(fd int32, offset int64, whence int32) int64
func exits(msg *byte)
func brk_(addr unsafe.Pointer) uintptr
func sleep(ms int32) int32
func rfork(flags int32, stk, mm, gg, fn unsafe.Pointer) int32
func plan9_semacquire(addr *uint32, block int32) int32
func plan9_tsemacquire(addr *uint32, ms int32) int32
func plan9_semrelease(addr *uint32, count int32) int32
func notify(fn unsafe.Pointer) int32
func noted(mode int32) int32
func nsec(*int64) int64
func sigtramp(ureg, msg unsafe.Pointer)
func setfpmasks()
func errstr() string

// The size of the note handler frame varies among architectures,
// but 512 bytes should be enough for every implementation.
const stackSystem = 512
