// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package runtime

import "unsafe"

func sbrk0() uintptr

// Called from write_err_android.go only, but defined in sys_linux_*.s;
// declared here (instead of in write_err_android.go) for go vet on non-android builds.
// The return value is the raw syscall result, which may encode an error number.
//go:noescape
func access(name *byte, mode int32) int32
func connect(fd int32, addr unsafe.Pointer, len int32) int32
func socket(domain int32, typ int32, prot int32) int32
