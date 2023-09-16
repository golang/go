// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
#include "_cgo_export.h"

__stdcall
static unsigned int
addThread(void *p)
{
	int i, max;
	
	max = *(int*)p;
	for(i=0; i<max; i++)
		Add(i);
	return 0;
}

void
doAdd(int max, int nthread)
{
	enum { MaxThread = 20 };
	int i;
	uintptr_t thread_id[MaxThread];
	
	if(nthread > MaxThread)
		nthread = MaxThread;
	for(i=0; i<nthread; i++)
		thread_id[i] = _beginthreadex(0, 0, addThread, &max, 0, 0);
	for(i=0; i<nthread; i++) {
		WaitForSingleObject((HANDLE)thread_id[i], INFINITE);
		CloseHandle((HANDLE)thread_id[i]);
	}
}

__stdcall
static unsigned int
goDummyCallbackThread(void* p)
{
	int i, max;

	max = *(int*)p;
	for(i=0; i<max; i++)
		goDummy();
	return 0;
}

int
callGoInCThread(int max)
{
	uintptr_t thread_id;
	thread_id = _beginthreadex(0, 0, goDummyCallbackThread, &max, 0, 0);
	WaitForSingleObject((HANDLE)thread_id, INFINITE);
	CloseHandle((HANDLE)thread_id);
	return max;
}
