// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>

int
runcmd(char **argv)
{
	int pid;
	Waitmsg *w;
	
	switch(pid = fork()) {
	case -1:
		return -1;
	case 0:
		exec(argv[0], argv);
		fprint(2, "exec %s: %r\n", argv[0]);
		exits("exec");
	}
	
	w = wait();
	if(w == nil)
		return -1;
	if(w->pid != pid) {
		werrstr("unexpected pid in wait");
		free(w);
		return -1;
	}
	if(w->msg[0]) {
		werrstr("unsuccessful exit status: %s", w->msg);
		free(w);
		return -1;
	}
	free(w);
	return 0;
}
