/*
Plan 9 from User Space src/lib9/create.c
http://code.swtch.com/plan9port/src/tip/src/lib9/create.c

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
#include <unistd.h>
#include <fcntl.h>
#include <libc.h>
#include <sys/stat.h>
#ifndef O_DIRECT
#define O_DIRECT 0
#endif

int
p9create(char *path, int mode, ulong perm)
{
	int fd, cexec, umode, rclose, lock, rdwr;
	struct flock fl;

	rdwr = mode&3;
	lock = mode&OLOCK;
	cexec = mode&OCEXEC;
	rclose = mode&ORCLOSE;
	mode &= ~(ORCLOSE|OCEXEC|OLOCK);

	/* XXX should get mode mask right? */
	fd = -1;
	if(perm&DMDIR){
		if(mode != OREAD){
			werrstr("bad mode in directory create");
			goto out;
		}
		if(mkdir(path, perm&0777) < 0)
			goto out;
		fd = open(path, O_RDONLY);
	}else{
		umode = (mode&3)|O_CREAT|O_TRUNC;
		mode &= ~(3|OTRUNC);
		if(mode&ODIRECT){
			umode |= O_DIRECT;
			mode &= ~ODIRECT;
		}
		if(mode&OEXCL){
			umode |= O_EXCL;
			mode &= ~OEXCL;
		}
		if(mode&OAPPEND){
			umode |= O_APPEND;
			mode &= ~OAPPEND;
		}
		if(mode){
			werrstr("unsupported mode in create");
			goto out;
		}
		fd = open(path, umode, perm);
	}
out:
	if(fd >= 0){
		if(lock){
			fl.l_type = (rdwr==OREAD) ? F_RDLCK : F_WRLCK;
			fl.l_whence = SEEK_SET;
			fl.l_start = 0;
			fl.l_len = 0;
			if(fcntl(fd, F_SETLK, &fl) < 0){
				close(fd);
				werrstr("lock: %r");
				return -1;
			}
		}
		if(cexec)
			fcntl(fd, F_SETFL, FD_CLOEXEC);
		if(rclose)
			remove(path);
	}
	return fd;
}
