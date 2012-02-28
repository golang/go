// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * objdump simulation - only enough to make pprof work on Macs
 */

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <mach.h>

void
usage(void)
{
	fprint(2, "usage: objdump binary start stop\n");
	fprint(2, "Disassembles binary from PC start up to stop.\n");
	exits("usage");
}

void
main(int argc, char **argv)
{
	int fd, n;
	uvlong pc, start, stop;
	Fhdr fhdr;
	Biobuf bout;
	char buf[1024];
	Map *text;

	ARGBEGIN{
	default:
		usage();
	}ARGEND

	if(argc != 3)
		usage();
	start = strtoull(argv[1], 0, 16);
	stop = strtoull(argv[2], 0, 16);

	fd = open(argv[0], OREAD);
	if(fd < 0)
		sysfatal("open %s: %r", argv[0]);
	if(crackhdr(fd, &fhdr) <= 0)
		sysfatal("crackhdr: %r");
	machbytype(fhdr.type);
	if(syminit(fd, &fhdr) <= 0)
		sysfatal("syminit: %r");
	text = loadmap(nil, fd, &fhdr);
	if(text == nil)
		sysfatal("loadmap: %r");

	Binit(&bout, 1, OWRITE);
	for(pc=start; pc<stop; ) {
		if(fileline(buf, sizeof buf, pc))
			Bprint(&bout, "%s\n", buf);
		buf[0] = '\0';
		machdata->das(text, pc, 0, buf, sizeof buf);
		Bprint(&bout, " %llx: %s\n", pc, buf);
		n = machdata->instsize(text, pc);
		if(n <= 0)
			break;
		pc += n;
	}
	Bflush(&bout);
	exits(0);
}
