// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

extern void *runtime·LoadLibrary;
extern void *runtime·GetProcAddress;

// Call a Windows function with stdcall conventions,
// and switch to os stack during the call.
#pragma	varargck	countpos	runtime·stdcall	2
#pragma	varargck	type		runtime·stdcall	void*
#pragma	varargck	type		runtime·stdcall	uintptr
void runtime·asmstdcall(void *c);
void *runtime·stdcall(void *fn, int32 count, ...);

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
#define	NSIG	65
