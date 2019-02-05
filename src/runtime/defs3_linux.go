// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

/*
Input to cgo -cdefs

GOARCH=ppc64 cgo -cdefs defs_linux.go defs3_linux.go > defs_linux_ppc64.h
*/

package runtime

/*
#define size_t __kernel_size_t
#define sigset_t __sigset_t // rename the sigset_t here otherwise cgo will complain about "inconsistent definitions for C.sigset_t"
#define	_SYS_TYPES_H	// avoid inclusion of sys/types.h
#include <asm/ucontext.h>
#include <asm-generic/fcntl.h>
*/
import "C"

const (
	O_RDONLY    = C.O_RDONLY
	O_CLOEXEC   = C.O_CLOEXEC
	SA_RESTORER = 0 // unused
)

type Usigset C.__sigset_t

// types used in sigcontext
type Ptregs C.struct_pt_regs
type Gregset C.elf_gregset_t
type FPregset C.elf_fpregset_t
type Vreg C.elf_vrreg_t

type StackT C.stack_t

// PPC64 uses sigcontext in place of mcontext in ucontext.
// see https://git.kernel.org/cgit/linux/kernel/git/torvalds/linux.git/tree/arch/powerpc/include/uapi/asm/ucontext.h
type Sigcontext C.struct_sigcontext
type Ucontext C.struct_ucontext
