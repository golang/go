// +build !plan9

/*
Plan 9 from User Space src/lib9/open.c
http://code.swtch.com/plan9port/src/tip/src/lib9/open.c

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

#define _GNU_SOURCE	/* for Linux O_DIRECT */
#include <u.h>
#define NOPLAN9DEFINES
#include <sys/file.h>
#include <libc.h>
#ifndef O_DIRECT
#define O_DIRECT 0
#endif

int
p9open(char *name, int mode)
{
	int rclose;
	int fd, umode, rdwr;

	rdwr = mode&3;
	umode = rdwr;
	rclose = mode&ORCLOSE;
	mode &= ~(3|ORCLOSE);
	if(mode&OTRUNC){
		umode |= O_TRUNC;
		mode ^= OTRUNC;
	}
	if(mode&ODIRECT){
		umode |= O_DIRECT;
		mode ^= ODIRECT;
	}
	if(mode&OAPPEND){
		umode |= O_APPEND;
		mode ^= OAPPEND;
	}
	if(mode){
		werrstr("mode 0x%x not supported", mode);
		return -1;
	}
	umode |= O_BINARY;
	fd = open(name, umode);
	if(fd >= 0){
		if(rclose)
			remove(name);
	}
	return fd;
}
