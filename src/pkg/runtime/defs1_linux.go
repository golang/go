// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

/*
Input to cgo -cdefs

GOARCH=amd64 cgo -cdefs defs.go defs1.go >amd64/defs.h
*/

package runtime

/*
#include <ucontext.h>
#include <fcntl.h>
*/
import "C"

const (
	O_RDONLY  = C.O_RDONLY
	O_CLOEXEC = C.O_CLOEXEC
)

type Usigset C.__sigset_t
type Fpxreg C.struct__libc_fpxreg
type Xmmreg C.struct__libc_xmmreg
type Fpstate C.struct__libc_fpstate
type Fpxreg1 C.struct__fpxreg
type Xmmreg1 C.struct__xmmreg
type Fpstate1 C.struct__fpstate
type Fpreg1 C.struct__fpreg
type SigaltstackT C.struct_sigaltstack
type Mcontext C.mcontext_t
type Ucontext C.ucontext_t
type Sigcontext C.struct_sigcontext
