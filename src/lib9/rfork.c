// +build !windows

/*
Plan 9 from User Space src/lib9/rfork.c
http://code.swtch.com/plan9port/src/tip/src/lib9/rfork.c

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
#include <sys/wait.h>
#include <signal.h>
#include <libc.h>
#undef rfork

static void
nop(int x)
{
	USED(x);
}

int
p9rfork(int flags)
{
	int pid, status;
	int p[2];
	int n;
	char buf[128], *q;
	extern char **environ;

	if((flags&(RFPROC|RFFDG|RFMEM)) == (RFPROC|RFFDG)){
		/* check other flags before we commit */
		flags &= ~(RFPROC|RFFDG|RFENVG);
		n = (flags & ~(RFNOTEG|RFNAMEG|RFNOWAIT|RFCENVG));
		if(n){
			werrstr("unknown flags %08ux in rfork", n);
			return -1;
		}
		if(flags&RFNOWAIT){
			/*
			 * BUG - should put the signal handler back after we
			 * finish, but I just don't care.  If a program calls with
			 * NOWAIT once, they're not likely to want child notes
			 * after that.
			 */
			signal(SIGCHLD, nop);
			if(pipe(p) < 0)
				return -1;
		}
		pid = fork();
		if(pid == -1)
			return -1;
		if(flags&RFNOWAIT){
			flags &= ~RFNOWAIT;
			if(pid){
				/*
				 * Parent - wait for child to fork wait-free child.
				 * Then read pid from pipe.  Assume pipe buffer can absorb the write.
				 */
				close(p[1]);
				status = 0;
				if(wait4(pid, &status, 0, 0) < 0){
					werrstr("pipe dance - wait4 - %r");
					close(p[0]);
					return -1;
				}
				n = readn(p[0], buf, sizeof buf-1);
				close(p[0]);
				if(!WIFEXITED(status) || WEXITSTATUS(status)!=0 || n <= 0){
					if(!WIFEXITED(status))
						werrstr("pipe dance - !exited 0x%ux", status);
					else if(WEXITSTATUS(status) != 0)
						werrstr("pipe dance - non-zero status 0x%ux", status);
					else if(n < 0)
						werrstr("pipe dance - pipe read error - %r");
					else if(n == 0)
						werrstr("pipe dance - pipe read eof");
					else
						werrstr("pipe dance - unknown failure");
					return -1;
				}
				buf[n] = 0;
				if(buf[0] == 'x'){
					werrstr("%s", buf+2);
					return -1;
				}
				pid = strtol(buf, &q, 0);
			}else{
				/*
				 * Child - fork a new child whose wait message can't
				 * get back to the parent because we're going to exit!
				 */
				signal(SIGCHLD, SIG_IGN);
				close(p[0]);
				pid = fork();
				if(pid){
					/* Child parent - send status over pipe and exit. */
					if(pid > 0)
						fprint(p[1], "%d", pid);
					else
						fprint(p[1], "x %r");
					close(p[1]);
					_exit(0);
				}else{
					/* Child child - close pipe. */
					close(p[1]);
				}
			}
		}
		if(pid != 0)
			return pid;
		if(flags&RFCENVG)
			if(environ)
				*environ = nil;
	}
	if(flags&RFPROC){
		werrstr("cannot use rfork for shared memory -- use libthread");
		return -1;
	}
	if(flags&RFNAMEG){
		/* XXX set $NAMESPACE to a new directory */
		flags &= ~RFNAMEG;
	}
	if(flags&RFNOTEG){
		setpgid(0, getpid());
		flags &= ~RFNOTEG;
	}
	if(flags&RFNOWAIT){
		werrstr("cannot use RFNOWAIT without RFPROC");
		return -1;
	}
	if(flags){
		werrstr("unknown flags %08ux in rfork", flags);
		return -1;
	}
	return 0;
}
