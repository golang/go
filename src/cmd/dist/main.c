// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "a.h"

int vflag;
char *argv0;

// cmdtab records the available commands.
static struct {
	char *name;
	void (*f)(int, char**);
} cmdtab[] = {
	{"banner", cmdbanner},
	{"bootstrap", cmdbootstrap},
	{"clean", cmdclean},
	{"env", cmdenv},
	{"install", cmdinstall},
	{"version", cmdversion},
};

// The OS-specific main calls into the portable code here.
void
xmain(int argc, char **argv)
{
	int i;

	if(argc <= 1)
		usage();
	
	for(i=0; i<nelem(cmdtab); i++) {
		if(streq(cmdtab[i].name, argv[1])) {
			cmdtab[i].f(argc-1, argv+1);
			return;
		}
	}

	xprintf("unknown command %s\n", argv[1]);
	usage();
}
