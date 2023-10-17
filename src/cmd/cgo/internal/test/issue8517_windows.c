// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "windows.h"

extern void testHandleLeaksCallback();

DWORD WINAPI testHandleLeaksFunc(LPVOID lpThreadParameter)
{
	int i;
	for(i = 0; i < 100; i++) {
		testHandleLeaksCallback();
	}
	return 0;
}

void testHandleLeaks()
{
	HANDLE h;
	h = CreateThread(NULL, 0, &testHandleLeaksFunc, 0, 0, NULL);
	WaitForSingleObject(h, INFINITE);
	CloseHandle(h);
}
