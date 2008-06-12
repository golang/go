/*
Plan 9 from User Space src/lib9/execl.c
http://code.swtch.com/plan9port/src/tip/src/lib9/execl.c

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

int
execl(char *prog, ...)
{
	int i;
	va_list arg;
	char **argv;

	va_start(arg, prog);
	for(i=0; va_arg(arg, char*) != nil; i++)
		;
	va_end(arg);

	argv = malloc((i+1)*sizeof(char*));
	if(argv == nil)
		return -1;

	va_start(arg, prog);
	for(i=0; (argv[i] = va_arg(arg, char*)) != nil; i++)
		;
	va_end(arg);

	exec(prog, argv);
	free(argv);
	return -1;
}

