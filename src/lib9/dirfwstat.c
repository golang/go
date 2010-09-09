/*
Plan 9 from User Space src/lib9/dirfwstat.c
http://code.swtch.com/plan9port/src/tip/src/lib9/dirfwstat.c

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

#define NOPLAN9DEFINES
#include <u.h>
#include <libc.h>
#include <sys/time.h>
#include <sys/stat.h>

#if defined(__FreeBSD__) || defined(__APPLE__) || defined(__OpenBSD__) || defined(__linux__)
/* do nothing -- futimes exists and is fine */

#elif defined(__SunOS5_9__)
/* use futimesat */
static int
futimes(int fd, struct timeval *tv)
{
	return futimesat(fd, 0, tv);
}

#else
/* provide dummy */
/* rename just in case -- linux provides an unusable one */
#undef futimes
#define futimes myfutimes
static int
futimes(int fd, struct timeval *tv)
{
	werrstr("futimes not available");
	return -1;
}

#endif

int
dirfwstat(int fd, Dir *dir)
{
	int ret;
	struct timeval tv[2];

	ret = 0;
#ifndef _WIN32
	if(~dir->mode != 0){
		if(fchmod(fd, dir->mode) < 0)
			ret = -1;
	}
#endif
	if(~dir->mtime != 0){
		tv[0].tv_sec = dir->mtime;
		tv[0].tv_usec = 0;
		tv[1].tv_sec = dir->mtime;
		tv[1].tv_usec = 0;
		if(futimes(fd, tv) < 0)
			ret = -1;
	}
	return ret;
}

