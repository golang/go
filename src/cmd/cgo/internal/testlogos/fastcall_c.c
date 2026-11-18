// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <time.h>
#include <errno.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <poll.h>
#include <unistd.h>
#endif

int
fast_add(int a, int b)
{
	return a + b;
}

int
slow_noescape(int a, int b)
{
	return a + b;
}

int
slow_nocallback(int a, int b)
{
	return a + b;
}

static int
do_sleep(int micros)
{
#ifdef _WIN32
	if (micros > 0) {
		Sleep((DWORD)((micros + 999) / 1000));
	}
#else
	struct timespec ts;
	ts.tv_sec = micros / 1000000;
	ts.tv_nsec = (long)(micros % 1000000) * 1000L;
	nanosleep(&ts, 0);
#endif
	return micros;
}

int
block_sleep(int micros)
{
	return do_sleep(micros);
}

int
plain_sleep(int micros)
{
	return do_sleep(micros);
}

int
read_int(int *p)
{
	return *p;
}

int
sum_bytes(unsigned char *p, int n)
{
	int s = 0;
	for (int i = 0; i < n; i++) {
		s += p[i];
	}
	return s;
}

static unsigned long long
do_spin(unsigned long long iters)
{
	unsigned long long x = 0;
	for (unsigned long long i = 0; i < iters; i++) {
		x += i * 6364136223846793005ULL + 1442695040888963407ULL;
		// Keep the loop observable to the CPU profiler: clang folds the
		// affine accumulation into a closed form otherwise, so busy_spin
		// returns without spending cycles.
		__asm__ __volatile__("" : "+r" (x));
	}
	return x;
}

unsigned long long
busy_spin(unsigned long long iters)
{
	return do_spin(iters);
}

unsigned long long
plain_spin(unsigned long long iters)
{
	return do_spin(iters);
}

#ifdef _WIN32

// Windows anonymous pipes are not waitable handles, so the gate polls
// PeekNamedPipe on a 10 ms tick against the same 30-second deadline the POSIX
// side gives poll.
static int
gate_signal_and_await(uintptr_t ready_h, uintptr_t gate_h)
{
	char c = 0;
	DWORD n;
	if (!WriteFile((HANDLE)ready_h, &c, 1, &n, 0) || n != 1) {
		return 1;
	}
	ULONGLONG deadline = GetTickCount64() + 30000;
	for (;;) {
		DWORD avail = 0;
		if (PeekNamedPipe((HANDLE)gate_h, 0, 0, 0, &avail, 0) && avail > 0) {
			break;
		}
		if (GetTickCount64() >= deadline) {
			return 2;
		}
		Sleep(10);
	}
	if (!ReadFile((HANDLE)gate_h, &c, 1, &n, 0) || n != 1) {
		return 3;
	}
	return 0;
}

int
gate_wait(uintptr_t ready_fd, uintptr_t gate_fd)
{
	return gate_signal_and_await(ready_fd, gate_fd);
}

int
gate_watch(uintptr_t ready_fd, uintptr_t gate_fd, unsigned char *p, unsigned char want)
{
	int r = gate_signal_and_await(ready_fd, gate_fd);
	if (r != 0) {
		return r;
	}
	if (*p != want) {
		return 4;
	}
	return 0;
}

#else

int
gate_wait(uintptr_t ready_fd, uintptr_t gate_fd)
{
	char c = 0;
	ssize_t n;
	do {
		n = write((int)ready_fd, &c, 1);
	} while (n < 0 && errno == EINTR);
	if (n != 1) {
		return 1;
	}
	struct pollfd pfd;
	pfd.fd = (int)gate_fd;
	pfd.events = POLLIN;
	int r;
	do {
		r = poll(&pfd, 1, 30000);
	} while (r < 0 && errno == EINTR);
	if (r <= 0) {
		return 2;
	}
	do {
		n = read((int)gate_fd, &c, 1);
	} while (n < 0 && errno == EINTR);
	if (n != 1) {
		return 3;
	}
	return 0;
}

int
gate_watch(uintptr_t ready_fd, uintptr_t gate_fd, unsigned char *p, unsigned char want)
{
	char c = 0;
	ssize_t n;
	do {
		n = write((int)ready_fd, &c, 1);
	} while (n < 0 && errno == EINTR);
	if (n != 1) {
		return 1;
	}
	struct pollfd pfd;
	pfd.fd = (int)gate_fd;
	pfd.events = POLLIN;
	int r;
	do {
		r = poll(&pfd, 1, 30000);
	} while (r < 0 && errno == EINTR);
	if (r <= 0) {
		return 2;
	}
	do {
		n = read((int)gate_fd, &c, 1);
	} while (n < 0 && errno == EINTR);
	if (n != 1) {
		return 3;
	}
	if (*p != want) {
		return 4;
	}
	return 0;
}

#endif
