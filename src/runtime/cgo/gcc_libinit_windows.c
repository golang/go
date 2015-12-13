// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo
#define WIN64_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>

#include <stdio.h>
#include <stdlib.h>

static volatile long runtime_init_once_gate = 0;
static volatile long runtime_init_once_done = 0;

static CRITICAL_SECTION runtime_init_cs;

static HANDLE runtime_init_wait;
static int runtime_init_done;

// Pre-initialize the runtime synchronization objects
void
_cgo_preinit_init() {
	 runtime_init_wait = CreateEvent(NULL, TRUE, FALSE, NULL);
	 if (runtime_init_wait == NULL) {
		fprintf(stderr, "runtime: failed to create runtime initialization wait event.\n");
		abort();
	 }

	 InitializeCriticalSection(&runtime_init_cs);
}

// Make sure that the preinit sequence has run.
void
_cgo_maybe_run_preinit() {
	 if (!InterlockedExchangeAdd(&runtime_init_once_done, 0)) {
			if (InterlockedIncrement(&runtime_init_once_gate) == 1) {
				 _cgo_preinit_init();
				 InterlockedIncrement(&runtime_init_once_done);
			} else {
				 // Decrement to avoid overflow.
				 InterlockedDecrement(&runtime_init_once_gate);
				 while(!InterlockedExchangeAdd(&runtime_init_once_done, 0)) {
						Sleep(0);
				 }
			}
	 }
}

void
x_cgo_sys_thread_create(void (*func)(void*), void* arg) {
	uintptr_t thandle;

	thandle = _beginthread(func, 0, arg);
	if(thandle == -1) {
		fprintf(stderr, "runtime: failed to create new OS thread (%d)\n", errno);
		abort();
	}
}

int
_cgo_is_runtime_initialized() {
	 EnterCriticalSection(&runtime_init_cs);
	 int status = runtime_init_done;
	 LeaveCriticalSection(&runtime_init_cs);
	 return status;
}

void
_cgo_wait_runtime_init_done() {
	 _cgo_maybe_run_preinit();
	while (!_cgo_is_runtime_initialized()) {
			WaitForSingleObject(runtime_init_wait, INFINITE);
	}
}

void
x_cgo_notify_runtime_init_done(void* dummy) {
	 _cgo_maybe_run_preinit();

	 EnterCriticalSection(&runtime_init_cs);
	runtime_init_done = 1;
	 LeaveCriticalSection(&runtime_init_cs);

	 if (!SetEvent(runtime_init_wait)) {
		fprintf(stderr, "runtime: failed to signal runtime initialization complete.\n");
		abort();
	}
}

