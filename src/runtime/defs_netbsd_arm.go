// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

/*
Input to cgo.

GOARCH=arm go tool cgo -cdefs defs_netbsd.go defs_netbsd_arm.go >defs_netbsd_arm.h
*/

package runtime

/*
#include <sys/types.h>
#include <machine/mcontext.h>
*/
import "C"

const (
	REG_R0   = C._REG_R0
	REG_R1   = C._REG_R1
	REG_R2   = C._REG_R2
	REG_R3   = C._REG_R3
	REG_R4   = C._REG_R4
	REG_R5   = C._REG_R5
	REG_R6   = C._REG_R6
	REG_R7   = C._REG_R7
	REG_R8   = C._REG_R8
	REG_R9   = C._REG_R9
	REG_R10  = C._REG_R10
	REG_R11  = C._REG_R11
	REG_R12  = C._REG_R12
	REG_R13  = C._REG_R13
	REG_R14  = C._REG_R14
	REG_R15  = C._REG_R15
	REG_CPSR = C._REG_CPSR
)
