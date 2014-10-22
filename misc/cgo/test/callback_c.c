// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <sys/types.h>
#include <unistd.h>
#include "_cgo_export.h"

void
callback(void *f)
{
	// use some stack space
	volatile char data[64*1024];

	data[0] = 0;
	goCallback(f);
        data[sizeof(data)-1] = 0;
}

void
callGoFoo(void)
{
	extern void goFoo(void);
	goFoo();
}

void
IntoC(void)
{
	BackIntoGo();
}

#ifdef WIN32
#include <windows.h>
long long
mysleep(int seconds) {
	long long st = GetTickCount();
	sleep(seconds);
	return st;
}
#else
#include <sys/time.h>
long long
mysleep(int seconds) {
	long long st;
	struct timeval tv;
	gettimeofday(&tv, NULL);
	st = tv.tv_sec * 1000 + tv.tv_usec / 1000;
	sleep(seconds);
	return st;
}
#endif

long long
twoSleep(int n)
{
	BackgroundSleep(n);
	return mysleep(n);
}

void
callGoStackCheck(void)
{
	extern void goStackCheck(void);
	goStackCheck();
}

int
returnAfterGrow(void)
{
	extern int goReturnVal(void);
	goReturnVal();
	return 123456;
}

int
returnAfterGrowFromGo(void)
{
	extern int goReturnVal(void);
	return goReturnVal();
}

