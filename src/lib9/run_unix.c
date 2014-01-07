// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

#include <u.h>
#include <errno.h>
#include <sys/wait.h>
#define NOPLAN9DEFINES
#include <libc.h>

int
runcmd(char **argv)
{
	int pid, pid1, status;
	
	switch(pid = fork()) {
	case -1:
		return -1;
	case 0:
		execvp(argv[0], argv);
		fprint(2, "exec %s: %r\n", argv[0]);
		_exit(1);
	}
	
	while((pid1 = wait(&status)) < 0) {
		if(errno != EINTR) {
			werrstr("waitpid: %r");
			return -1;
		}
	}
	if(pid1 != pid) {
		werrstr("unexpected pid in wait");
		return -1;
	}
	if(!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
		werrstr("unsuccessful exit status %#x", status);
		return -1;
	}
	return 0;
}

