// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix && !openbsd

#include <assert.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <ucontext.h>

// Use a stack size larger than the 32kb estimate in
// runtime.callbackUpdateSystemStack. This ensures that a second stack
// allocation won't accidentally count as in bounds of the first stack
#define STACK_SIZE	(64ull << 10)

static ucontext_t uctx_save, uctx_switch;

extern void stackSwitchCallback(void);

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
	char *stack2 = malloc(STACK_SIZE);
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
	free(stack2);

	return NULL;
}

void callStackSwitchCallbackFromThread(void) {
	pthread_t thread;
	assert(pthread_create(&thread, NULL, stackSwitchThread, NULL) == 0);
	assert(pthread_join(thread, NULL) == 0);
}
