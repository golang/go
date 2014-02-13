// +build !plan9

/*
Plan 9 from User Space src/lib9/errstr.c
http://code.swtch.com/plan9port/src/tip/src/lib9/errstr.c

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

/*
 * We assume there's only one error buffer for the whole system.
 * If you use ffork, you need to provide a _syserrstr.  Since most
 * people will use libthread (which provides a _syserrstr), this is
 * okay.
 */

#include <u.h>
#include <errno.h>
#include <libc.h>

enum
{
	EPLAN9 = 0x19283745
};

char *(*_syserrstr)(void);
static char xsyserr[ERRMAX];
static char*
getsyserr(void)
{
	char *s;

	s = nil;
	if(_syserrstr)
		s = (*_syserrstr)();
	if(s == nil)
		s = xsyserr;
	return s;
}

int
errstr(char *err, uint n)
{
	char tmp[ERRMAX];
	char *syserr;

	strecpy(tmp, tmp+ERRMAX, err);
	rerrstr(err, n);
	syserr = getsyserr();
	strecpy(syserr, syserr+ERRMAX, tmp);
	errno = EPLAN9;
	return 0;
}

void
rerrstr(char *err, uint n)
{
	char *syserr;

	syserr = getsyserr();
	if(errno == EINTR)
		strcpy(syserr, "interrupted");
	else if(errno != EPLAN9)
		strcpy(syserr, strerror(errno));
	strecpy(err, err+n, syserr);
}

/* replaces __errfmt in libfmt */

int
__errfmt(Fmt *f)
{
	if(errno == EPLAN9)
		return fmtstrcpy(f, getsyserr());
	return fmtstrcpy(f, strerror(errno));
}

void
werrstr(char *fmt, ...)
{
	va_list arg;
	char buf[ERRMAX];

	va_start(arg, fmt);
	vseprint(buf, buf+ERRMAX, fmt, arg);
	va_end(arg);
	errstr(buf, ERRMAX);
}

