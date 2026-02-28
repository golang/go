// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix && !android && !openbsd

// Required for darwin ucontext.
#define _XOPEN_SOURCE
// Required for netbsd stack_t if _XOPEN_SOURCE is set.
#define _XOPEN_SOURCE_EXTENDED	1
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <assert.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <ucontext.h>

// musl libc does not provide getcontext, etc. Skip the test there.
//
// musl libc doesn't provide any direct detection mechanism. So assume any
// non-glibc linux is using musl.
//
// Note that bionic does not provide getcontext either, but that is skipped via
// the android build tag.
#if defined(__linux__) && !defined(__GLIBC__)
#define MUSL 1
#endif
#if defined(MUSL)
void callStackSwitchCallbackFromThread(void) {
	printf("SKIP\n");
	exit(0);
}
#else

// Use a stack size larger than the 32kb estimate in
// runtime.callbackUpdateSystemStack. This ensures that a second stack
// allocation won't accidentally count as in bounds of the first stack
#define STACK_SIZE	(64ull << 10)

static ucontext_t uctx_save, uctx_switch;

extern void stackSwitchCallback(void);

char *stack2;

static void *stackSwitchThread(void *arg) {
	// Simple test: callback works from the normal system stack.
	stackSwitchCallback();

	// Next, verify that switching stacks doesn't break callbacks.

	char *stack1 = malloc(STACK_SIZE);
	if (stack1 == NULL) {
		perror("malloc");
		exit(1);
	}

	// Allocate the second stack before freeing the first to ensure we don't get
	// the same address from malloc.
	//
	// Will be freed in stackSwitchThread2.
	stack2 = malloc(STACK_SIZE);
	if (stack1 == NULL) {
		perror("malloc");
		exit(1);
	}

	if (getcontext(&uctx_switch) == -1) {
		perror("getcontext");
		exit(1);
	}
	uctx_switch.uc_stack.ss_sp = stack1;
	uctx_switch.uc_stack.ss_size = STACK_SIZE;
	uctx_switch.uc_link = &uctx_save;
	makecontext(&uctx_switch, stackSwitchCallback, 0);

	if (swapcontext(&uctx_save, &uctx_switch) == -1) {
		perror("swapcontext");
		exit(1);
	}

	if (getcontext(&uctx_switch) == -1) {
		perror("getcontext");
		exit(1);
	}
	uctx_switch.uc_stack.ss_sp = stack2;
	uctx_switch.uc_stack.ss_size = STACK_SIZE;
	uctx_switch.uc_link = &uctx_save;
	makecontext(&uctx_switch, stackSwitchCallback, 0);

	if (swapcontext(&uctx_save, &uctx_switch) == -1) {
		perror("swapcontext");
		exit(1);
	}

	free(stack1);

	return NULL;
}

static void *stackSwitchThread2(void *arg) {
	// New thread. Use stack bounds that partially overlap the previous
	// bounds. needm should refresh the stack bounds anyway since this is a
	// new thread.

	// N.B. since we used a custom stack with makecontext,
	// callbackUpdateSystemStack had to guess the bounds. Its guess assumes
	// a 32KiB stack.
	char *prev_stack_lo = stack2 + STACK_SIZE - (32*1024);

	// New SP is just barely in bounds, but if we don't update the bounds
	// we'll almost certainly overflow. The SP that
	// callbackUpdateSystemStack sees already has some data pushed, so it
	// will be a bit below what we set here. Thus we include some slack.
	char *new_stack_hi = prev_stack_lo + 128;

	if (getcontext(&uctx_switch) == -1) {
		perror("getcontext");
		exit(1);
	}
	uctx_switch.uc_stack.ss_sp = new_stack_hi - (STACK_SIZE / 2);
	uctx_switch.uc_stack.ss_size = STACK_SIZE / 2;
	uctx_switch.uc_link = &uctx_save;
	makecontext(&uctx_switch, stackSwitchCallback, 0);

	if (swapcontext(&uctx_save, &uctx_switch) == -1) {
		perror("swapcontext");
		exit(1);
	}

	free(stack2);

	return NULL;
}

void callStackSwitchCallbackFromThread(void) {
	pthread_t thread;
	assert(pthread_create(&thread, NULL, stackSwitchThread, NULL) == 0);
	assert(pthread_join(thread, NULL) == 0);

	assert(pthread_create(&thread, NULL, stackSwitchThread2, NULL) == 0);
	assert(pthread_join(thread, NULL) == 0);
}

#endif
