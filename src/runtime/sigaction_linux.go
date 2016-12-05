// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !amd64

package runtime

// rt_sigaction calls the rt_sigaction system call. It is implemented in assembly.
//go:noescape
func rt_sigaction(sig uintptr, new, old *sigactiont, size uintptr) int32
