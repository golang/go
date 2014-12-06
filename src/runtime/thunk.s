// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file exposes various internal runtime functions to other packages in std lib.

#include "textflag.h"

#ifdef GOARCH_arm
#define JMP B
#endif
#ifdef GOARCH_ppc64
#define JMP BR
#endif
#ifdef GOARCH_ppc64le
#define JMP BR
#endif

TEXT net·runtimeNano(SB),NOSPLIT,$0-0
	JMP	runtime·nanotime(SB)

TEXT time·runtimeNano(SB),NOSPLIT,$0-0
	JMP     runtime·nanotime(SB)

TEXT time·Sleep(SB),NOSPLIT,$0-0
	JMP     runtime·timeSleep(SB)

TEXT time·startTimer(SB),NOSPLIT,$0-0
	JMP     runtime·startTimer(SB)

TEXT time·stopTimer(SB),NOSPLIT,$0-0
	JMP     runtime·stopTimer(SB)

TEXT sync·runtime_Syncsemacquire(SB),NOSPLIT,$0-0
	JMP	runtime·syncsemacquire(SB)

TEXT sync·runtime_Syncsemrelease(SB),NOSPLIT,$0-0
	JMP	runtime·syncsemrelease(SB)

TEXT sync·runtime_Syncsemcheck(SB),NOSPLIT,$0-0
	JMP	runtime·syncsemcheck(SB)

TEXT sync·runtime_Semacquire(SB),NOSPLIT,$0-0
	JMP	runtime·asyncsemacquire(SB)

TEXT sync·runtime_Semrelease(SB),NOSPLIT,$0-0
	JMP	runtime·asyncsemrelease(SB)

TEXT sync·runtime_registerPoolCleanup(SB),NOSPLIT,$0-0
	JMP	runtime·registerPoolCleanup(SB)

TEXT net·runtime_Semacquire(SB),NOSPLIT,$0-0
	JMP	runtime·asyncsemacquire(SB)

TEXT net·runtime_Semrelease(SB),NOSPLIT,$0-0
	JMP	runtime·asyncsemrelease(SB)

TEXT runtime∕pprof·runtime_cyclesPerSecond(SB),NOSPLIT,$0-0
	JMP	runtime·tickspersecond(SB)

TEXT bytes·Compare(SB),NOSPLIT,$0-0
	JMP	runtime·cmpbytes(SB)

TEXT reflect·call(SB), NOSPLIT, $0-0
	JMP	runtime·reflectcall(SB)

TEXT reflect·chanclose(SB), NOSPLIT, $0-0
	JMP	runtime·closechan(SB)

TEXT reflect·chanlen(SB), NOSPLIT, $0-0
	JMP	runtime·reflect_chanlen(SB)

TEXT reflect·chancap(SB), NOSPLIT, $0-0
	JMP	runtime·reflect_chancap(SB)

TEXT reflect·chansend(SB), NOSPLIT, $0-0
	JMP	runtime·reflect_chansend(SB)

TEXT reflect·chanrecv(SB), NOSPLIT, $0-0
	JMP	runtime·reflect_chanrecv(SB)

TEXT reflect·memmove(SB), NOSPLIT, $0-0
	JMP	runtime·memmove(SB)

TEXT runtime∕debug·freeOSMemory(SB), NOSPLIT, $0-0
	JMP	runtime·freeOSMemory(SB)

TEXT runtime∕debug·WriteHeapDump(SB), NOSPLIT, $0-0
	JMP	runtime·writeHeapDump(SB)

TEXT net·runtime_pollServerInit(SB),NOSPLIT,$0-0
	JMP	runtime·netpollServerInit(SB)

TEXT net·runtime_pollOpen(SB),NOSPLIT,$0-0
	JMP	runtime·netpollOpen(SB)

TEXT net·runtime_pollClose(SB),NOSPLIT,$0-0
	JMP	runtime·netpollClose(SB)

TEXT net·runtime_pollReset(SB),NOSPLIT,$0-0
	JMP	runtime·netpollReset(SB)

TEXT net·runtime_pollWait(SB),NOSPLIT,$0-0
	JMP	runtime·netpollWait(SB)

TEXT net·runtime_pollWaitCanceled(SB),NOSPLIT,$0-0
	JMP	runtime·netpollWaitCanceled(SB)

TEXT net·runtime_pollSetDeadline(SB),NOSPLIT,$0-0
	JMP	runtime·netpollSetDeadline(SB)

TEXT net·runtime_pollUnblock(SB),NOSPLIT,$0-0
	JMP	runtime·netpollUnblock(SB)

TEXT syscall·setenv_c(SB), NOSPLIT, $0-0
	JMP	runtime·syscall_setenv_c(SB)

TEXT syscall·unsetenv_c(SB), NOSPLIT, $0-0
	JMP	runtime·syscall_unsetenv_c(SB)

TEXT reflect·makemap(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_makemap(SB)

TEXT reflect·mapaccess(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_mapaccess(SB)

TEXT reflect·mapassign(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_mapassign(SB)

TEXT reflect·mapdelete(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_mapdelete(SB)

TEXT reflect·mapiterinit(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_mapiterinit(SB)

TEXT reflect·mapiterkey(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_mapiterkey(SB)

TEXT reflect·mapiternext(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_mapiternext(SB)

TEXT reflect·maplen(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_maplen(SB)

TEXT reflect·ismapkey(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_ismapkey(SB)

TEXT reflect·ifaceE2I(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_ifaceE2I(SB)

TEXT reflect·unsafe_New(SB),NOSPLIT,$0-0
	JMP	runtime·newobject(SB)

TEXT reflect·unsafe_NewArray(SB),NOSPLIT,$0-0
	JMP	runtime·newarray(SB)

TEXT reflect·makechan(SB),NOSPLIT,$0-0
	JMP	runtime·makechan(SB)

TEXT reflect·rselect(SB),NOSPLIT,$0-0
	JMP	runtime·reflect_rselect(SB)

TEXT os·sigpipe(SB),NOSPLIT,$0-0
	JMP	runtime·os_sigpipe(SB)

TEXT runtime·runtime_init(SB),NOSPLIT,$0-0
	JMP	runtime·init(SB)

TEXT runtime·main_init(SB),NOSPLIT,$0-0
	JMP	main·init(SB)

TEXT runtime·main_main(SB),NOSPLIT,$0-0
	JMP	main·main(SB)

TEXT runtime·timenow(SB),NOSPLIT,$0-0
	JMP	time·now(SB)

TEXT sync∕atomic·runtime_procPin(SB),NOSPLIT,$0-0
	JMP     sync·runtime_procPin(SB)

TEXT sync∕atomic·runtime_procUnpin(SB),NOSPLIT,$0-0
	JMP     sync·runtime_procUnpin(SB)

TEXT syscall·runtime_envs(SB),NOSPLIT,$0-0
	JMP	runtime·runtime_envs(SB)

TEXT os·runtime_args(SB),NOSPLIT,$0-0
	JMP	runtime·runtime_args(SB)

TEXT sync·runtime_procUnpin(SB),NOSPLIT,$0-0
	JMP	runtime·sync_procUnpin(SB)

TEXT sync·runtime_procPin(SB),NOSPLIT,$0-0
	JMP	runtime·sync_procPin(SB)

TEXT syscall·runtime_BeforeFork(SB),NOSPLIT,$0-0
	JMP	runtime·syscall_BeforeFork(SB)

TEXT syscall·runtime_AfterFork(SB),NOSPLIT,$0-0
	JMP	runtime·syscall_AfterFork(SB)

TEXT reflect·typelinks(SB),NOSPLIT,$0-0
	JMP	runtime·typelinks(SB)
