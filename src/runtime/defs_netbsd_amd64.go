// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

/*
Input to cgo.

GOARCH=amd64 go tool cgo -cdefs defs_netbsd.go defs_netbsd_amd64.go >defs_netbsd_amd64.h
*/

package runtime

/*
#include <sys/types.h>
#include <machine/mcontext.h>
*/
import "C"

const (
	REG_RDI    = C._REG_RDI
	REG_RSI    = C._REG_RSI
	REG_RDX    = C._REG_RDX
	REG_RCX    = C._REG_RCX
	REG_R8     = C._REG_R8
	REG_R9     = C._REG_R9
	REG_R10    = C._REG_R10
	REG_R11    = C._REG_R11
	REG_R12    = C._REG_R12
	REG_R13    = C._REG_R13
	REG_R14    = C._REG_R14
	REG_R15    = C._REG_R15
	REG_RBP    = C._REG_RBP
	REG_RBX    = C._REG_RBX
	REG_RAX    = C._REG_RAX
	REG_GS     = C._REG_GS
	REG_FS     = C._REG_FS
	REG_ES     = C._REG_ES
	REG_DS     = C._REG_DS
	REG_TRAPNO = C._REG_TRAPNO
	REG_ERR    = C._REG_ERR
	REG_RIP    = C._REG_RIP
	REG_CS     = C._REG_CS
	REG_RFLAGS = C._REG_RFLAGS
	REG_RSP    = C._REG_RSP
	REG_SS     = C._REG_SS
)
