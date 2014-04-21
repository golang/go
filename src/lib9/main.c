// +build !plan9

/*
Plan 9 from User Space src/lib9/main.c
http://code.swtch.com/plan9port/src/tip/src/lib9/main.c

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

#ifdef WIN32
#include <windows.h>

static void
crashhandler(int sig)
{
	USED(sig);
	fprint(2, "%s: internal fatal error.\n", argv0);
	exit(1);
}
#endif

extern void p9main(int, char**);

int
main(int argc, char **argv)
{
#ifdef WIN32
	signal(SIGSEGV, crashhandler);
	signal(SIGBUS, crashhandler);
	// don't display the crash dialog
	DWORD mode = SetErrorMode(SEM_NOGPFAULTERRORBOX);
	SetErrorMode(mode | SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
#endif
	argv0 = argv[0];
	p9main(argc, argv);
	exits("main");
	return 99;
}
