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
	FILE *fin;
	char buf[1024], *p, *q;

	if(argc != 2) {
		fprintf(stderr, "usage: mksys sys.6\n");
		exit(1);
	}
	if((fin = fopen(argv[1], "r")) == NULL) {
		fprintf(stderr, "open %s: %s\n", argv[1], strerror(errno));
		exit(1);
	}

	// look for $$ that introduces imports
	while(fgets(buf, sizeof buf, fin) != NULL)
		if(strstr(buf, "$$"))
			goto begin;
	fprintf(stderr, "did not find beginning of imports\n");
	exit(1);

begin:
	printf("char *sysimport = \n");

	// process imports, stopping at $$ that closes them
	while(fgets(buf, sizeof buf, fin) != NULL) {
		buf[strlen(buf)-1] = 0;	// chop \n
		if(strstr(buf, "$$"))
			goto end;

		// chop leading white space
		for(p=buf; *p==' ' || *p == '\t'; p++)
			;

		// cut out decl of init_sys_function - it doesn't exist
		if(strstr(buf, "init_sys_function"))
			continue;

		// sys.go claims to be in package SYS to avoid
		// conflicts during "6g sys.go".  rename SYS to sys.
		for(q=p; *q; q++)
			if(memcmp(q, "SYS", 3) == 0)
				memmove(q, "sys", 3);

		printf("\t\"%s\\n\"\n", p);
	}
	fprintf(stderr, "did not find end of imports\n");
	exit(1);

end:
	printf("\t\"$$\\n\";\n");
	return 0;
}
