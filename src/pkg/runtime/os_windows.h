// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

extern void *runtime·LoadLibrary;
extern void *runtime·GetProcAddress;
extern void *runtime·GetQueuedCompletionStatusEx;

// Call a Windows function with stdcall conventions,
// and switch to os stack during the call.
void runtime·asmstdcall(void *c);
void *runtime·stdcall0(void *fn);
void *runtime·stdcall1(void *fn, uintptr a0);
void *runtime·stdcall2(void *fn, uintptr a0, uintptr a1);
void *runtime·stdcall3(void *fn, uintptr a0, uintptr a1, uintptr a2);
void *runtime·stdcall4(void *fn, uintptr a0, uintptr a1, uintptr a2, uintptr a3);
void *runtime·stdcall5(void *fn, uintptr a0, uintptr a1, uintptr a2, uintptr a3, uintptr a4);
void *runtime·stdcall6(void *fn, uintptr a0, uintptr a1, uintptr a2, uintptr a3, uintptr a4, uintptr a5);
void *runtime·stdcall7(void *fn, uintptr a0, uintptr a1, uintptr a2, uintptr a3, uintptr a4, uintptr a5, uintptr a6);

uint32 runtime·getlasterror(void);
void runtime·setlasterror(uint32 err);

// Function to be called by windows CreateThread
// to start new os thread.
uint32 runtime·tstart_stdcall(M *newm);

uint32 runtime·issigpanic(uint32);
void runtime·sigpanic(void);
uint32 runtime·ctrlhandler(uint32 type);

// Windows dll function to go callback entry.
byte *runtime·compilecallback(Eface fn, bool cleanstack);
void *runtime·callbackasm(void);

void runtime·install_exception_handler(void);
void runtime·remove_exception_handler(void);

// TODO(brainman): should not need those
enum {
	NSIG = 65,
};
