// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The following function allows one to dynamically
// resolve DLL function names.
// The arguments are strings.
void *get_proc_addr(void *library, void *name);

extern void *VirtualAlloc;
extern void *LoadLibraryEx;
extern void *GetProcAddress;
extern void *GetLastError;

#define goargs windows_goargs
void windows_goargs(void);

// Get start address of symbol data in memory.
void *get_symdat_addr(void);

// Call a Windows function with stdcall conventions.
void *stdcall_raw(void *fn, ...);

// Call a Windows function with stdcall conventions,
// and switch to os stack during the call.
void *stdcall(void *fn, int32 count, ...);

// Call stdcall Windows function StdcallParams.fn
// with params StdcallParams.args,
// followed immediately by GetLastError call.
// Both return values are returned in StdcallParams.r and
// StdcallParams.err. Will use os stack during the call.
typedef struct StdcallParams StdcallParams;
struct StdcallParams
{
	void	*fn;
	uintptr args[12];
	uintptr	r;
	uintptr	err;
};
void call_syscall(void *args);
void syscall(StdcallParams *p);
