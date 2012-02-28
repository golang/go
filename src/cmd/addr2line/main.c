// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * addr2line simulation - only enough to make pprof work on Macs
 */

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <mach.h>

void
usage(void)
{
	fprint(2, "usage: addr2line binary\n");
	fprint(2, "reads addresses from standard input and writes two lines for each:\n");
	fprint(2, "\tfunction name\n");
	fprint(2, "\tfile:line\n");
	exits("usage");
}

void
main(int argc, char **argv)
{
	int fd;
	char *p;
	uvlong pc;
	Symbol s;
	Fhdr fhdr;
	Biobuf bin, bout;
	char file[1024];

	ARGBEGIN{
	default:
		usage();
	}ARGEND

	if(argc != 1)
		usage();

	fd = open(argv[0], OREAD);
	if(fd < 0)
		sysfatal("open %s: %r", argv[0]);
	if(crackhdr(fd, &fhdr) <= 0)
		sysfatal("crackhdr: %r");
	machbytype(fhdr.type);
	if(syminit(fd, &fhdr) <= 0)
		sysfatal("syminit: %r");

	Binit(&bin, 0, OREAD);
	Binit(&bout, 1, OWRITE);
	for(;;) {
		p = Brdline(&bin, '\n');
		if(p == nil)
			break;
		p[Blinelen(&bin)-1] = '\0';
		pc = strtoull(p, 0, 16);
		if(!findsym(pc, CTEXT, &s))
			s.name = "??";
		if(!fileline(file, sizeof file, pc))
			strcpy(file, "??:0");
		Bprint(&bout, "%s\n%s\n", s.name, file);
	}
	Bflush(&bout);
	exits(0);
}
