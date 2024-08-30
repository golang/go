// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The windows C definitions for trace.go. That file uses //export so
// it can't put function definitions in the "C" import comment.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
#include "_cgo_export.h"

extern void goCalledFromC(void);
extern void goCalledFromCThread(void);

__stdcall
static unsigned int cCalledFromCThread(void *p) {
	goCalledFromCThread();
	return 0;
}

void cCalledFromGo(void) {
	goCalledFromC();

	uintptr_t thread;
	thread = _beginthreadex(NULL, 0, cCalledFromCThread, NULL, 0, NULL);
	WaitForSingleObject((HANDLE)thread, INFINITE);
	CloseHandle((HANDLE)thread);
}
