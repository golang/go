// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Extract import data from sys.6 and generate C string version.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>

int
main(int argc, char **argv)
{
	char *name;
	FILE *fin;
	char buf[1024], initfunc[1024], *p, *q;

	if(argc != 2) {
		fprintf(stderr, "usage: sys sys\n");
		fprintf(stderr, "in file $1.6 s/PACKAGE/$1/\n");
		exit(1);
	}

	name = argv[1];
	snprintf(initfunc, sizeof(initfunc), "init_%s_function", name);

	snprintf(buf, sizeof(buf), "%s.6", name);
	if((fin = fopen(buf, "r")) == NULL) {
		fprintf(stderr, "open %s: %s\n", buf, strerror(errno));
		exit(1);
	}

	// look for $$ that introduces imports
	while(fgets(buf, sizeof buf, fin) != NULL)
		if(strstr(buf, "$$"))
			goto begin;
	fprintf(stderr, "did not find beginning of imports\n");
	exit(1);

begin:
	printf("char *%simport = \n", name);

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
		while(q = strstr(p, "PACKAGE")) {
			*q = 0;
			printf("%s", p);	// up to the substitution
			printf("%s", name);	// the sub name
			p = q+7;		// continue with rest
		}

		printf("%s\\n\"\n", p);
	}
	fprintf(stderr, "did not find end of imports\n");
	exit(1);

end:
	printf("\t\"$$\\n\";\n");
	return 0;
}
