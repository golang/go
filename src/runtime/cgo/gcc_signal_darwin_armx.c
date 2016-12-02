// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Emulation of the Unix signal SIGSEGV.
//
// On iOS, Go tests and apps under development are run by lldb.
// The debugger uses a task-level exception handler to intercept signals.
// Despite having a 'handle' mechanism like gdb, lldb will not allow a
// SIGSEGV to pass to the running program. For Go, this means we cannot
// generate a panic, which cannot be recovered, and so tests fail.
//
// We work around this by registering a thread-level mach exception handler
// and intercepting EXC_BAD_ACCESS. The kernel offers thread handlers a
// chance to resolve exceptions before the task handler, so we can generate
// the panic and avoid lldb's SIGSEGV handler.
//
// The dist tool enables this by build flag when testing.

// +build lldb
// +build darwin
// +build arm arm64

#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#include <mach/arm/thread_status.h>
#include <mach/exception_types.h>
#include <mach/mach.h>
#include <mach/mach_init.h>
#include <mach/mach_port.h>
#include <mach/thread_act.h>
#include <mach/thread_status.h>

#include "libcgo.h"
#include "libcgo_unix.h"

uintptr_t x_cgo_panicmem;

static pthread_mutex_t mach_exception_handler_port_set_mu;
static mach_port_t mach_exception_handler_port_set = MACH_PORT_NULL;

kern_return_t
catch_exception_raise(
	mach_port_t exception_port,
	mach_port_t thread,
	mach_port_t task,
	exception_type_t exception,
	exception_data_t code_vector,
	mach_msg_type_number_t code_count)
{
	kern_return_t ret;
	arm_unified_thread_state_t thread_state;
	mach_msg_type_number_t state_count = ARM_UNIFIED_THREAD_STATE_COUNT;

	// Returning KERN_SUCCESS intercepts the exception.
	//
	// Returning KERN_FAILURE lets the exception fall through to the
	// next handler, which is the standard signal emulation code
	// registered on the task port.

	if (exception != EXC_BAD_ACCESS) {
		return KERN_FAILURE;
	}

	ret = thread_get_state(thread, ARM_UNIFIED_THREAD_STATE, (thread_state_t)&thread_state, &state_count);
	if (ret) {
		fprintf(stderr, "runtime/cgo: thread_get_state failed: %d\n", ret);
		abort();
	}

	// Bounce call to sigpanic through asm that makes it look like
	// we call sigpanic directly from the faulting code.
#ifdef __arm64__
	thread_state.ts_64.__x[1] = thread_state.ts_64.__lr;
	thread_state.ts_64.__x[2] = thread_state.ts_64.__pc;
	thread_state.ts_64.__pc = x_cgo_panicmem;
#else
	thread_state.ts_32.__r[1] = thread_state.ts_32.__lr;
	thread_state.ts_32.__r[2] = thread_state.ts_32.__pc;
	thread_state.ts_32.__pc = x_cgo_panicmem;
#endif

	if (0) {
		// Useful debugging logic when panicmem is broken.
		//
		// Sends the first SIGSEGV and lets lldb catch the
		// second one, avoiding a loop that locks up iOS
		// devices requiring a hard reboot.
		fprintf(stderr, "runtime/cgo: caught exc_bad_access\n");
		fprintf(stderr, "__lr = %llx\n", thread_state.ts_64.__lr);
		fprintf(stderr, "__pc = %llx\n", thread_state.ts_64.__pc);
		static int pass1 = 0;
		if (pass1) {
			return KERN_FAILURE;
		}
		pass1 = 1;
	}

	ret = thread_set_state(thread, ARM_UNIFIED_THREAD_STATE, (thread_state_t)&thread_state, state_count);
	if (ret) {
		fprintf(stderr, "runtime/cgo: thread_set_state failed: %d\n", ret);
		abort();
	}

	return KERN_SUCCESS;
}

void
darwin_arm_init_thread_exception_port()
{
	// Called by each new OS thread to bind its EXC_BAD_ACCESS exception
	// to mach_exception_handler_port_set.
	int ret;
	mach_port_t port = MACH_PORT_NULL;

	ret = mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &port);
	if (ret) {
		fprintf(stderr, "runtime/cgo: mach_port_allocate failed: %d\n", ret);
		abort();
	}
	ret = mach_port_insert_right(
		mach_task_self(),
		port,
		port,
		MACH_MSG_TYPE_MAKE_SEND);
	if (ret) {
		fprintf(stderr, "runtime/cgo: mach_port_insert_right failed: %d\n", ret);
		abort();
	}

	ret = thread_set_exception_ports(
		mach_thread_self(),
		EXC_MASK_BAD_ACCESS,
		port,
		EXCEPTION_DEFAULT,
		THREAD_STATE_NONE);
	if (ret) {
		fprintf(stderr, "runtime/cgo: thread_set_exception_ports failed: %d\n", ret);
		abort();
	}

	ret = pthread_mutex_lock(&mach_exception_handler_port_set_mu);
	if (ret) {
		fprintf(stderr, "runtime/cgo: pthread_mutex_lock failed: %d\n", ret);
		abort();
	}
	ret = mach_port_move_member(
		mach_task_self(),
		port,
		mach_exception_handler_port_set);
	if (ret) {
		fprintf(stderr, "runtime/cgo: mach_port_move_member failed: %d\n", ret);
		abort();
	}
	ret = pthread_mutex_unlock(&mach_exception_handler_port_set_mu);
	if (ret) {
		fprintf(stderr, "runtime/cgo: pthread_mutex_unlock failed: %d\n", ret);
		abort();
	}
}

static void*
mach_exception_handler(void *port)
{
	// Calls catch_exception_raise.
	extern boolean_t exc_server();
	mach_msg_server(exc_server, 2048, (mach_port_t)port, 0);
	abort(); // never returns
}

void
darwin_arm_init_mach_exception_handler()
{
	pthread_mutex_init(&mach_exception_handler_port_set_mu, NULL);

	// Called once per process to initialize a mach port server, listening
	// for EXC_BAD_ACCESS thread exceptions.
	int ret;
	pthread_t thr = NULL;
	pthread_attr_t attr;
	sigset_t ign, oset;

	ret = mach_port_allocate(
		mach_task_self(),
		MACH_PORT_RIGHT_PORT_SET,
		&mach_exception_handler_port_set);
	if (ret) {
		fprintf(stderr, "runtime/cgo: mach_port_allocate failed for port_set: %d\n", ret);
		abort();
	}

	// Block all signals to the exception handler thread
	sigfillset(&ign);
	pthread_sigmask(SIG_SETMASK, &ign, &oset);

	// Start a thread to handle exceptions.
	uintptr_t port_set = (uintptr_t)mach_exception_handler_port_set;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	ret = _cgo_try_pthread_create(&thr, &attr, mach_exception_handler, (void*)port_set);

	pthread_sigmask(SIG_SETMASK, &oset, nil);

	if (ret) {
		fprintf(stderr, "runtime/cgo: pthread_create failed: %d\n", ret);
		abort();
	}
	pthread_attr_destroy(&attr);
}
