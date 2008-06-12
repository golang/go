// Derived from Inferno libmach/map.c and
// Plan 9 from User Space src/libmach/map.c
//
// http://code.swtch.com/plan9port/src/tip/src/libmach/map.c
// http://code.google.com/p/inferno-os/source/browse/utils/libmach/map.c
//
//
//	Copyright © 1994-1999 Lucent Technologies Inc.
//	Power PC support Copyright © 1995-2004 C H Forsyth (forsyth@terzarima.net).
//	Portions Copyright © 1997-1999 Vita Nuova Limited.
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com).
//	Revisions Copyright © 2000-2004 Lucent Technologies Inc. and others.
//	Portions Copyright © 2001-2007 Russ Cox.
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/*
 * file map routines
 */
#include <u.h>
#include <libc.h>
#include <bio.h>
#include <mach_amd64.h>

Map *
newmap(Map *map, int n)
{
	int size;

	size = sizeof(Map)+(n-1)*sizeof(struct segment);
	if (map == 0)
		map = malloc(size);
	else
		map = realloc(map, size);
	if (map == 0) {
		werrstr("out of memory: %r");
		return 0;
	}
	memset(map, 0, size);
	map->nsegs = n;
	return map;
}

int
setmap(Map *map, int fd, uvlong b, uvlong e, vlong f, char *name)
{
	int i;

	if (map == 0)
		return 0;
	for (i = 0; i < map->nsegs; i++)
		if (!map->seg[i].inuse)
			break;
	if (i >= map->nsegs)
		return 0;
	map->seg[i].b = b;
	map->seg[i].e = e;
	map->seg[i].f = f;
	map->seg[i].inuse = 1;
	map->seg[i].name = name;
	map->seg[i].fd = fd;
	return 1;
}

static uvlong
stacktop(int pid)
{
	char buf[64];
	int fd;
	int n;
	char *cp;

	snprint(buf, sizeof(buf), "/proc/%d/segment", pid);
	fd = open(buf, 0);
	if (fd < 0)
		return 0;
	n = read(fd, buf, sizeof(buf)-1);
	close(fd);
	buf[n] = 0;
	if (strncmp(buf, "Stack", 5))
		return 0;
	for (cp = buf+5; *cp && *cp == ' '; cp++)
		;
	if (!*cp)
		return 0;
	cp = strchr(cp, ' ');
	if (!cp)
		return 0;
	while (*cp && *cp == ' ')
		cp++;
	if (!*cp)
		return 0;
	return strtoull(cp, 0, 16);
}

Map*
attachproc(int pid, int kflag, int corefd, Fhdr *fp)
{
	char buf[64], *regs;
	int fd;
	Map *map;
	uvlong n;
	int mode;

	map = newmap(0, 4);
	if (!map)
		return 0;
	if(kflag) {
		regs = "kregs";
		mode = OREAD;
	} else {
		regs = "regs";
		mode = ORDWR;
	}
	if (mach->regsize) {
		sprint(buf, "/proc/%d/%s", pid, regs);
		fd = open(buf, mode);
		if(fd < 0) {
			free(map);
			return 0;
		}
		setmap(map, fd, 0, mach->regsize, 0, "regs");
	}
	if (mach->fpregsize) {
		sprint(buf, "/proc/%d/fpregs", pid);
		fd = open(buf, mode);
		if(fd < 0) {
			close(map->seg[0].fd);
			free(map);
			return 0;
		}
		setmap(map, fd, mach->regsize, mach->regsize+mach->fpregsize, 0, "fpregs");
	}
	setmap(map, corefd, fp->txtaddr, fp->txtaddr+fp->txtsz, fp->txtaddr, "text");
	if(kflag || fp->dataddr >= mach->utop) {
		setmap(map, corefd, fp->dataddr, ~0, fp->dataddr, "data");
		return map;
	}
	n = stacktop(pid);
	if (n == 0) {
		setmap(map, corefd, fp->dataddr, mach->utop, fp->dataddr, "data");
		return map;
	}
	setmap(map, corefd, fp->dataddr, n, fp->dataddr, "data");
	return map;
}
	
int
findseg(Map *map, char *name)
{
	int i;

	if (!map)
		return -1;
	for (i = 0; i < map->nsegs; i++)
		if (map->seg[i].inuse && !strcmp(map->seg[i].name, name))
			return i;
	return -1;
}

void
unusemap(Map *map, int i)
{
	if (map != 0 && 0 <= i && i < map->nsegs)
		map->seg[i].inuse = 0;
}

Map*
loadmap(Map *map, int fd, Fhdr *fp)
{
	map = newmap(map, 2);
	if (map == 0)
		return 0;

	map->seg[0].b = fp->txtaddr;
	map->seg[0].e = fp->txtaddr+fp->txtsz;
	map->seg[0].f = fp->txtoff;
	map->seg[0].fd = fd;
	map->seg[0].inuse = 1;
	map->seg[0].name = "text";
	map->seg[1].b = fp->dataddr;
	map->seg[1].e = fp->dataddr+fp->datsz;
	map->seg[1].f = fp->datoff;
	map->seg[1].fd = fd;
	map->seg[1].inuse = 1;
	map->seg[1].name = "data";
	return map;
}
