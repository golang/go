// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

/*
Input to cgo.

GOARCH=amd64 go tool cgo -cdefs defs_solaris.go defs_solaris_amd64.go >defs_solaris_amd64.h
*/

package runtime

/*
#include <sys/types.h>
#include <sys/regset.h>
*/
import "C"

const (
	REG_RDI    = C.REG_RDI
	REG_RSI    = C.REG_RSI
	REG_RDX    = C.REG_RDX
	REG_RCX    = C.REG_RCX
	REG_R8     = C.REG_R8
	REG_R9     = C.REG_R9
	REG_R10    = C.REG_R10
	REG_R11    = C.REG_R11
	REG_R12    = C.REG_R12
	REG_R13    = C.REG_R13
	REG_R14    = C.REG_R14
	REG_R15    = C.REG_R15
	REG_RBP    = C.REG_RBP
	REG_RBX    = C.REG_RBX
	REG_RAX    = C.REG_RAX
	REG_GS     = C.REG_GS
	REG_FS     = C.REG_FS
	REG_ES     = C.REG_ES
	REG_DS     = C.REG_DS
	REG_TRAPNO = C.REG_TRAPNO
	REG_ERR    = C.REG_ERR
	REG_RIP    = C.REG_RIP
	REG_CS     = C.REG_CS
	REG_RFLAGS = C.REG_RFL
	REG_RSP    = C.REG_RSP
	REG_SS     = C.REG_SS
)
