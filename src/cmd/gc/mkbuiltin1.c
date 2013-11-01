// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// Compile .go file, import data from .6 file, and generate C string version.

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <stdarg.h>

void esc(char*);
void fatal(char*, ...);

int
main(int argc, char **argv)
{
	char *name;
	FILE *fin;
	char buf[1024], initfunc[1024], *p, *q;

	if(argc != 2) {
		fprintf(stderr, "usage: mkbuiltin1 sys\n");
		fatal("in file $1.6 s/PACKAGE/$1/");
	}

	name = argv[1];
	snprintf(initfunc, sizeof(initfunc), "init_%s_function", name);

	snprintf(buf, sizeof(buf), "%s.%s", name, getenv("O"));
	if((fin = fopen(buf, "r")) == NULL) {
		fatal("open %s: %s", buf, strerror(errno));
	}

	// look for $$ that introduces imports
	while(fgets(buf, sizeof buf, fin) != NULL)
		if(strstr(buf, "$$"))
			goto begin;
	fatal("did not find beginning of imports");

begin:
	printf("char *%simport =\n", name);

	// process imports, stopping at $$ that closes them
	while(fgets(buf, sizeof buf, fin) != NULL) {
		buf[strlen(buf)-1] = 0;	// chop \n
		if(strstr(buf, "$$"))
			goto end;

		// chop leading white space
		for(p=buf; *p==' ' || *p == '\t'; p++)
			;

		// cut out decl of init_$1_function - it doesn't exist
		if(strstr(buf, initfunc))
			continue;

		// sys.go claims to be in package PACKAGE to avoid
		// conflicts during "6g sys.go".  rename PACKAGE to $2.
		printf("\t\"");
		while((q = strstr(p, "PACKAGE")) != NULL) {
			*q = 0;
			esc(p);	// up to the substitution
			printf("%s", name);	// the sub name
			p = q+7;		// continue with rest
		}

		esc(p);
		printf("\\n\"\n");
	}
	fatal("did not find end of imports");

end:
	printf("\t\"$$\\n\";\n");
	return 0;
}

void
esc(char *p)
{
	for(; *p; p++) {
		if(*p == '\\' || *p == '\"')
			printf("\\");
		putchar(*p);
	}
}

void
fatal(char *msg, ...)
{
	va_list arg;
	
	va_start(arg, msg);
	fprintf(stderr, "fatal: ");
	vfprintf(stderr, msg, arg);
	fprintf(stderr, "\n");
	exit(2);
}
