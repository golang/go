// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

/*
Input to cgo.

GOARCH=386 go tool cgo -cdefs defs_netbsd.go defs_netbsd_386.go >defs_netbsd_386.h
*/

package runtime

/*
#include <sys/types.h>
#include <machine/mcontext.h>
*/
import "C"

const (
	REG_GS     = C._REG_GS
	REG_FS     = C._REG_FS
	REG_ES     = C._REG_ES
	REG_DS     = C._REG_DS
	REG_EDI    = C._REG_EDI
	REG_ESI    = C._REG_ESI
	REG_EBP    = C._REG_EBP
	REG_ESP    = C._REG_ESP
	REG_EBX    = C._REG_EBX
	REG_EDX    = C._REG_EDX
	REG_ECX    = C._REG_ECX
	REG_EAX    = C._REG_EAX
	REG_TRAPNO = C._REG_TRAPNO
	REG_ERR    = C._REG_ERR
	REG_EIP    = C._REG_EIP
	REG_CS     = C._REG_CS
	REG_EFL    = C._REG_EFL
	REG_UESP   = C._REG_UESP
	REG_SS     = C._REG_SS
)
