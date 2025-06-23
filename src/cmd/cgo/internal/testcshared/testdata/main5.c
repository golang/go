// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that a signal handler works in non-Go code when using
// os/signal.Notify.
// This is a lot like ../testcarchive/main3.c.

#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sched.h>
#include <dlfcn.h>

static void die(const char* msg) {
	perror(msg);
	exit(EXIT_FAILURE);
}

static volatile sig_atomic_t sigioSeen;

static void ioHandler(int signo, siginfo_t* info, void* ctxt) {
	sigioSeen = 1;
}

int main(int argc, char** argv) {
	int verbose;
	struct sigaction sa;
	void* handle;
	void (*catchSIGIO)(void);
	void (*resetSIGIO)(void);
	void (*awaitSIGIO)();
	bool (*sawSIGIO)();
	int i;
	struct timespec ts;

	verbose = argc > 2;
	setvbuf(stdout, NULL, _IONBF, 0);

	if (verbose) {
		fprintf(stderr, "calling sigaction\n");
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

	if (verbose) {
		fprintf(stderr, "calling dlopen\n");
	}

	handle = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
	if (handle == NULL) {
		fprintf(stderr, "%s\n", dlerror());
		exit(EXIT_FAILURE);
	}

	// At this point there should not be a Go signal handler
	// installed for SIGIO.

	if (verbose) {
		fprintf(stderr, "raising SIGIO\n");
	}

	if (raise(SIGIO) < 0) {
		die("raise");
	}

	if (verbose) {
		fprintf(stderr, "waiting for sigioSeen\n");
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

	sigioSeen = 0;

	// Tell the Go code to catch SIGIO.

	if (verbose) {
		fprintf(stderr, "calling dlsym\n");
	}

	catchSIGIO = (void(*)(void))dlsym(handle, "CatchSIGIO");
	if (catchSIGIO == NULL) {
		fprintf(stderr, "%s\n", dlerror());
		exit(EXIT_FAILURE);
	}

	if (verbose) {
		fprintf(stderr, "calling CatchSIGIO\n");
	}

	catchSIGIO();

	if (verbose) {
		fprintf(stderr, "raising SIGIO\n");
	}

	if (raise(SIGIO) < 0) {
		die("raise");
	}

	if (verbose) {
		fprintf(stderr, "calling dlsym\n");
	}

	// Check that the Go code saw SIGIO.
	awaitSIGIO = (void (*)(void))dlsym(handle, "AwaitSIGIO");
	if (awaitSIGIO == NULL) {
		fprintf(stderr, "%s\n", dlerror());
		exit(EXIT_FAILURE);
	}

	if (verbose) {
		fprintf(stderr, "calling AwaitSIGIO\n");
	}

	awaitSIGIO();

	if (sigioSeen != 0) {
		fprintf(stderr, "C handler saw SIGIO when only Go handler should have\n");
		exit(EXIT_FAILURE);
	}

	// Tell the Go code to stop catching SIGIO.

	if (verbose) {
		fprintf(stderr, "calling dlsym\n");
	}

	resetSIGIO = (void (*)(void))dlsym(handle, "ResetSIGIO");
	if (resetSIGIO == NULL) {
		fprintf(stderr, "%s\n", dlerror());
		exit(EXIT_FAILURE);
	}

	if (verbose) {
		fprintf(stderr, "calling ResetSIGIO\n");
	}

	resetSIGIO();

	sawSIGIO = (bool (*)(void))dlsym(handle, "SawSIGIO");
	if (sawSIGIO == NULL) {
		fprintf(stderr, "%s\n", dlerror());
		exit(EXIT_FAILURE);
	}

	if (verbose) {
		fprintf(stderr, "raising SIGIO\n");
	}

	if (raise(SIGIO) < 0) {
		die("raise");
	}

	if (verbose) {
		fprintf(stderr, "calling SawSIGIO\n");
	}

	if (sawSIGIO()) {
		fprintf(stderr, "Go handler saw SIGIO after Reset\n");
		exit(EXIT_FAILURE);
	}

	if (verbose) {
		fprintf(stderr, "waiting for sigioSeen\n");
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

	printf("PASS\n");
	return 0;
}
