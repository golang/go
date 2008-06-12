/*
Plan 9 from User Space src/lib9/sysfatal.c
http://code.swtch.com/plan9port/src/tip/src/lib9/sysfatal.c

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
#include <libc.h>

void (*_sysfatal)(char*, ...);

void
sysfatal(char *fmt, ...)
{
	char buf[256];
	va_list arg;

	va_start(arg, fmt);
	if(_sysfatal)
		(*_sysfatal)(fmt, arg);
	vseprint(buf, buf+sizeof buf, fmt, arg);
	va_end(arg);

	__fixargv0();
	fprint(2, "%s: %s\n", argv0 ? argv0 : "<prog>", buf);
	exits("fatal");
}

