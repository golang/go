// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test source is used by both TestBigStackCallbackCgo (linked
// directly into the Go binary) and TestBigStackCallbackSyscall
// (compiled into a DLL).

#include <windows.h>
#include <stdio.h>

#ifndef STACK_SIZE_PARAM_IS_A_RESERVATION
#define STACK_SIZE_PARAM_IS_A_RESERVATION 0x00010000
#endif

typedef void callback(char*);

// Allocate a stack that's much larger than the default.
static const int STACK_SIZE = 16<<20;

static callback *bigStackCallback;

static void useStack(int bytes) {
	// Windows doesn't like huge frames, so we grow the stack 64k at a time.
	char x[64<<10];
	if (bytes < sizeof x) {
		bigStackCallback(x);
	} else {
		useStack(bytes - sizeof x);
	}
}

static DWORD WINAPI threadEntry(LPVOID lpParam) {
	useStack(STACK_SIZE - (128<<10));
	return 0;
}

void bigStack(callback *cb) {
	bigStackCallback = cb;
	HANDLE hThread = CreateThread(NULL, STACK_SIZE, threadEntry, NULL, STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
	if (hThread == NULL) {
		fprintf(stderr, "CreateThread failed\n");
		exit(1);
	}
	WaitForSingleObject(hThread, INFINITE);
}
