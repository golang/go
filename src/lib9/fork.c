/*
Plan 9 from User Space src/lib9/fork.c
http://code.swtch.com/plan9port/src/tip/src/lib9/fork.c

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
#include <signal.h>
#include <libc.h>
#include "9proc.h"
#undef fork

int
p9fork(void)
{
	int pid;
	sigset_t all, old;

	sigfillset(&all);
	sigprocmask(SIG_SETMASK, &all, &old);
	pid = fork();
	if(pid == 0){
		_clearuproc();
		_p9uproc(0);
	}
	sigprocmask(SIG_SETMASK, &old, nil);
	return pid;
}
