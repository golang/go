// +build !plan9

/*
Plan 9 from User Space src/lib9/dirfstat.c
http://code.swtch.com/plan9port/src/tip/src/lib9/dirfstat.c

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
#define NOPLAN9DEFINES
#include <libc.h>

#include <sys/stat.h>

extern int _p9dir(struct stat*, struct stat*, char*, Dir*, char**, char*);

Dir*
dirfstat(int fd)
{
	struct stat st;
	int nstr;
	Dir *d;
	char *str, tmp[100];

	if(fstat(fd, &st) < 0)
		return nil;

	snprint(tmp, sizeof tmp, "/dev/fd/%d", fd);
	nstr = _p9dir(&st, &st, tmp, nil, nil, nil);
	d = malloc(sizeof(Dir)+(size_t)nstr);
	if(d == nil)
		return nil;
	memset(d, 0, sizeof(Dir)+(size_t)nstr);
	str = (char*)&d[1];
	_p9dir(&st, &st, tmp, d, &str, str+nstr);
	return d;
}

