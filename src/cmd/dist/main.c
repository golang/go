// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "a.h"

// cmdtab records the available commands.
static struct {
	char *name;
	void (*f)(int, char**);
} cmdtab[] = {
	{"bootstrap", cmdbootstrap},
	{"env", cmdenv},
	{"install", cmdinstall},
};

// The OS-specific main calls into the portable code here.
void
xmain(int argc, char **argv)
{
	int i;

	if(argc <= 1) {
		xprintf("go tool dist commands:\n");
		for(i=0; i<nelem(cmdtab); i++)
			xprintf("\t%s\n", cmdtab[i].name);
		xexit(1);
	}
	
	for(i=0; i<nelem(cmdtab); i++) {
		if(streq(cmdtab[i].name, argv[1])) {
			cmdtab[i].f(argc-1, argv+1);
			return;
		}
	}

	fatal("unknown command %s", argv[1]);
}
