// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that a signal handler that uses up stack space does not crash
// if the signal is delivered to a thread running a goroutine.
// This is a lot like misc/cgo/testcarchive/main2.c.

#include <setjmp.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <dlfcn.h>

static void die(const char* msg) {
	perror(msg);
	exit(EXIT_FAILURE);
}

static volatile sig_atomic_t sigioSeen;

// Use up some stack space.
static void recur(int i, char *p) {
	char a[1024];

	*p = '\0';
	if (i > 0) {
		recur(i - 1, a);
	}
}

// Signal handler that uses up more stack space than a goroutine will have.
static void ioHandler(int signo, siginfo_t* info, void* ctxt) {
	char a[1024];

	recur(4, a);
	sigioSeen = 1;
}

static jmp_buf jmp;
static char* nullPointer;

// Signal handler for SIGSEGV on a C thread.
static void segvHandler(int signo, siginfo_t* info, void* ctxt) {
	sigset_t mask;
	int i;

	if (sigemptyset(&mask) < 0) {
		die("sigemptyset");
	}
	if (sigaddset(&mask, SIGSEGV) < 0) {
		die("sigaddset");
	}
	i = sigprocmask(SIG_UNBLOCK, &mask, NULL);
	if (i != 0) {
		fprintf(stderr, "sigprocmask: %s\n", strerror(i));
		exit(EXIT_FAILURE);
	}

	// Don't try this at home.
	longjmp(jmp, signo);

	// We should never get here.
	abort();
}

int main(int argc, char** argv) {
	int verbose;
	struct sigaction sa;
	void* handle;
	void (*fn)(void);
	sigset_t mask;
	int i;
	struct timespec ts;

	verbose = argc > 2;
	setvbuf(stdout, NULL, _IONBF, 0);

	// Call setsid so that we can use kill(0, SIGIO) below.
	// Don't check the return value so that this works both from
	// a job control shell and from a shell script.
	setsid();

	if (verbose) {
		printf("calling sigaction\n");
	}

	memset(&sa, 0, sizeof sa);
	sa.sa_sigaction = ioHandler;
	if (sigemptyset(&sa.sa_mask) < 0) {
		die("sigemptyset");
	}
	sa.sa_flags = SA_SIGINFO;
	if (sigaction(SIGIO, &sa, NULL) < 0) {
		die("sigaction");
	}

	sa.sa_sigaction = segvHandler;
	if (sigaction(SIGSEGV, &sa, NULL) < 0 || sigaction(SIGBUS, &sa, NULL) < 0) {
		die("sigaction");
	}

	if (verbose) {
		printf("calling dlopen\n");
	}

	handle = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
	if (handle == NULL) {
		fprintf(stderr, "%s\n", dlerror());
		exit(EXIT_FAILURE);
	}

	if (verbose) {
		printf("calling dlsym\n");
	}

	// Start some goroutines.
	fn = (void(*)(void))dlsym(handle, "RunGoroutines");
	if (fn == NULL) {
		fprintf(stderr, "%s\n", dlerror());
		exit(EXIT_FAILURE);
	}

	if (verbose) {
		printf("calling RunGoroutines\n");
	}

	fn();

	// Block SIGIO in this thread to make it more likely that it
	// will be delivered to a goroutine.

	if (verbose) {
		printf("calling pthread_sigmask\n");
	}

	if (sigemptyset(&mask) < 0) {
		die("sigemptyset");
	}
	if (sigaddset(&mask, SIGIO) < 0) {
		die("sigaddset");
	}
	i = pthread_sigmask(SIG_BLOCK, &mask, NULL);
	if (i != 0) {
		fprintf(stderr, "pthread_sigmask: %s\n", strerror(i));
		exit(EXIT_FAILURE);
	}

	if (verbose) {
		printf("calling kill\n");
	}

	if (kill(0, SIGIO) < 0) {
		die("kill");
	}

	if (verbose) {
		printf("waiting for sigioSeen\n");
	}

	// Wait until the signal has been delivered.
	i = 0;
	while (!sigioSeen) {
		ts.tv_sec = 0;
		ts.tv_nsec = 1000000;
		nanosleep(&ts, NULL);
		i++;
		if (i > 5000) {
			fprintf(stderr, "looping too long waiting for signal\n");
			exit(EXIT_FAILURE);
		}
	}

	if (verbose) {
		printf("calling setjmp\n");
	}

	// Test that a SIGSEGV on this thread is delivered to us.
	if (setjmp(jmp) == 0) {
		if (verbose) {
			printf("triggering SIGSEGV\n");
		}

		*nullPointer = '\0';

		fprintf(stderr, "continued after address error\n");
		exit(EXIT_FAILURE);
	}

	if (verbose) {
		printf("calling dlsym\n");
	}

	// Make sure that a SIGSEGV in Go causes a run-time panic.
	fn = (void (*)(void))dlsym(handle, "TestSEGV");
	if (fn == NULL) {
		fprintf(stderr, "%s\n", dlerror());
		exit(EXIT_FAILURE);
	}

	if (verbose) {
		printf("calling TestSEGV\n");
	}

	fn();

	printf("PASS\n");
	return 0;
}
