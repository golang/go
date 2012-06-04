/*
Plan 9 from User Space src/lib9/getwd.c
http://code.swtch.com/plan9port/src/tip/src/lib9/getwd.c

Copyright 2001-2007 Russ Cox.  All Rights Reserved.
Portions Copyright 2011 The Go Authors.  All Rights Reserved.

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
#include <errno.h>
#include <sys/stat.h>
#define NOPLAN9DEFINES
#include <libc.h>

char*
p9getwd(char *s, int ns)
{
	char *pwd;
	struct stat st1, st2;

	// Clumsy but widespread kludge:
	// if $PWD is set and matches ".", use it.
	// Matches glibc's get_current_dir_name and Go's os.Getwd.
	pwd = getenv("PWD");  // note: getenv, not p9getenv, so no free
	if(pwd != nil && pwd[0] &&
			stat(pwd, &st1) >= 0 && stat(".", &st2) >= 0 &&
			st1.st_dev == st2.st_dev && st1.st_ino == st2.st_ino) {
		if(strlen(pwd) >= ns) {
			errno = ERANGE;
			return nil;
		}
		strcpy(s, pwd);
		return s;
	}

	return getcwd(s, ns);
}
