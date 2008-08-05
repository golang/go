//	Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <mach_amd64.h>

Map*	
attachproc(int pid, Fhdr *fp)
{
	sysfatal("attachproc not implemented");
	return nil;
}

int
ctlproc(int pid, char *msg)
{
	sysfatal("ctlproc not implemented");
	return -1;
}

void
detachproc(Map *m)
{
	sysfatal("detachproc not implemented");
}

int
procnotes(int pid, char ***pnotes)
{
	sysfatal("procnotes not implemented");
	return -1;
}

char*
proctextfile(int pid)
{
	sysfatal("proctextfile not implemented");
	return nil;
}

int	
procthreadpids(int pid, int **thread)
{
	sysfatal("procthreadpids not implemented");
	return -1;
}

