// +build !plan9

/*
Plan 9 from User Space src/lib9/time.c
http://code.swtch.com/plan9port/src/tip/src/lib9/time.c

Copyright 2001-2007 Russ Cox.  All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <u.h>
#include <sys/time.h>
#include <time.h>
#ifndef _WIN32
#include <sys/resource.h>
#endif
#define NOPLAN9DEFINES
#include <libc.h>

long
p9times(long *t)
{
#ifdef _WIN32
	memset(t, 0, 4*sizeof(long));
#else
	struct rusage ru, cru;

	if(getrusage(0, &ru) < 0 || getrusage(-1, &cru) < 0)
		return -1;

	t[0] = ru.ru_utime.tv_sec*1000 + ru.ru_utime.tv_usec/1000;
	t[1] = ru.ru_stime.tv_sec*1000 + ru.ru_stime.tv_usec/1000;
	t[2] = cru.ru_utime.tv_sec*1000 + cru.ru_utime.tv_usec/1000;
	t[3] = cru.ru_stime.tv_sec*1000 + cru.ru_stime.tv_usec/1000;
#endif

	/* BUG */
	return t[0]+t[1]+t[2]+t[3];
}

double
p9cputime(void)
{
	long t[4];
	double d;

	if(p9times(t) < 0)
		return -1.0;

	d = (double)t[0]+(double)t[1]+(double)t[2]+(double)t[3];
	return d/1000.0;
}
