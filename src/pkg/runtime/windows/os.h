// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

extern void *runtime·LoadLibraryEx;
extern void *runtime·GetProcAddress;
extern void *runtime·GetLastError;

// Get start address of symbol data in memory.
void *runtime·get_symdat_addr(void);

// Call a Windows function with stdcall conventions,
// and switch to os stack during the call.
void *runtime·stdcall_raw(void *fn, int32 count, uintptr *args);
void *runtime·stdcall(void *fn, int32 count, ...);

// Function to be called by windows CreateTread
// to start new os thread.
uint32 runtime·tstart_stdcall(M *newm);

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
	int32	n;
	uintptr	r;
	uintptr	err;
};

void runtime·syscall(StdcallParams *p);
