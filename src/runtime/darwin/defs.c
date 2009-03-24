// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Input to godefs.
 *
	godefs -f -m64 defs.c >amd64/defs.h
	godefs defs.c >386/defs.h
 */

#define __DARWIN_UNIX03 0

#include <mach/mach.h>
#include <mach/message.h>
#include <sys/types.h>
#include <sys/time.h>
#include <signal.h>
#include <sys/mman.h>

enum {
	$PROT_NONE = PROT_NONE,
	$PROT_READ = PROT_READ,
	$PROT_WRITE = PROT_WRITE,
	$PROT_EXEC = PROT_EXEC,

	$MAP_ANON = MAP_ANON,
	$MAP_PRIVATE = MAP_PRIVATE,

	$MACH_MSG_TYPE_MOVE_RECEIVE = MACH_MSG_TYPE_MOVE_RECEIVE,
	$MACH_MSG_TYPE_MOVE_SEND = MACH_MSG_TYPE_MOVE_SEND,
	$MACH_MSG_TYPE_MOVE_SEND_ONCE = MACH_MSG_TYPE_MOVE_SEND_ONCE,
	$MACH_MSG_TYPE_COPY_SEND = MACH_MSG_TYPE_COPY_SEND,
	$MACH_MSG_TYPE_MAKE_SEND = MACH_MSG_TYPE_MAKE_SEND,
	$MACH_MSG_TYPE_MAKE_SEND_ONCE = MACH_MSG_TYPE_MAKE_SEND_ONCE,
	$MACH_MSG_TYPE_COPY_RECEIVE = MACH_MSG_TYPE_COPY_RECEIVE,

	$MACH_MSG_PORT_DESCRIPTOR = MACH_MSG_PORT_DESCRIPTOR,
	$MACH_MSG_OOL_DESCRIPTOR = MACH_MSG_OOL_DESCRIPTOR,
	$MACH_MSG_OOL_PORTS_DESCRIPTOR = MACH_MSG_OOL_PORTS_DESCRIPTOR,
	$MACH_MSG_OOL_VOLATILE_DESCRIPTOR = MACH_MSG_OOL_VOLATILE_DESCRIPTOR,

	$MACH_MSGH_BITS_COMPLEX = MACH_MSGH_BITS_COMPLEX,

	$MACH_SEND_MSG = MACH_SEND_MSG,
	$MACH_RCV_MSG = MACH_RCV_MSG,
	$MACH_RCV_LARGE = MACH_RCV_LARGE,

	$MACH_SEND_TIMEOUT = MACH_SEND_TIMEOUT,
	$MACH_SEND_INTERRUPT = MACH_SEND_INTERRUPT,
	$MACH_SEND_CANCEL = MACH_SEND_CANCEL,
	$MACH_SEND_ALWAYS = MACH_SEND_ALWAYS,
	$MACH_SEND_TRAILER = MACH_SEND_TRAILER,
	$MACH_RCV_TIMEOUT = MACH_RCV_TIMEOUT,
	$MACH_RCV_NOTIFY = MACH_RCV_NOTIFY,
	$MACH_RCV_INTERRUPT = MACH_RCV_INTERRUPT,
	$MACH_RCV_OVERWRITE = MACH_RCV_OVERWRITE,

	$NDR_PROTOCOL_2_0 = NDR_PROTOCOL_2_0,
	$NDR_INT_BIG_ENDIAN = NDR_INT_BIG_ENDIAN,
	$NDR_INT_LITTLE_ENDIAN = NDR_INT_LITTLE_ENDIAN,
	$NDR_FLOAT_IEEE = NDR_FLOAT_IEEE,
	$NDR_CHAR_ASCII = NDR_CHAR_ASCII,

	$SA_SIGINFO = SA_SIGINFO,
	$SA_RESTART = SA_RESTART,
	$SA_ONSTACK = SA_ONSTACK,
	$SA_USERTRAMP = SA_USERTRAMP,
	$SA_64REGSET = SA_64REGSET,
};

typedef mach_msg_body_t	$MachBody;
typedef mach_msg_header_t	$MachHeader;
typedef NDR_record_t		$MachNDR;
typedef mach_msg_port_descriptor_t	$MachPort;

typedef stack_t	$StackT;
typedef union __sigaction_u	$Sighandler;

typedef struct __sigaction	$Sigaction;	// used in syscalls
// typedef struct sigaction	$Sigaction;	// used by the C library
typedef union sigval $Sigval;
typedef siginfo_t $Siginfo;

typedef struct fp_control $FPControl;
typedef struct fp_status $FPStatus;
typedef struct mmst_reg $RegMMST;
typedef struct xmm_reg $RegXMM;

#ifdef __LP64__
// amd64
typedef x86_thread_state64_t	$Regs;
typedef x86_float_state64_t $FloatState;
typedef x86_exception_state64_t $ExceptionState;
typedef struct mcontext64 $Mcontext;
#else
// 386
typedef x86_thread_state32_t	$Regs;
typedef x86_float_state32_t $FloatState;
typedef x86_exception_state32_t $ExceptionState;
typedef struct mcontext32 $Mcontext;
#endif

typedef ucontext_t	$Ucontext;
