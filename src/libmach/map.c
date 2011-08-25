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
#include <mach.h>

Map *
newmap(Map *map, int n)
{
	int size;

	size = sizeof(Map)+(n-1)*sizeof(Seg);
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
setmap(Map *map, int fd, uvlong b, uvlong e, vlong f, char *name, Maprw *rw)
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
	map->seg[i].rw = rw;
	return 1;
}

/*
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
*/

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

int
fdrw(Map *map, Seg *s, uvlong addr, void *v, uint n, int isread)
{
	int tot, m;
	
	USED(map);

	for(tot=0; tot<n; tot+=m){
		if(isread)
			m = pread(s->fd, (uchar*)v+tot, n-tot, addr+tot);
		else
			m = pwrite(s->fd, (uchar*)v+tot, n-tot, addr+tot);
		if(m == 0){
			werrstr("short %s", isread ? "read" : "write");
			return -1;
		}
		if(m < 0){
			werrstr("%s %d at %#llux (+%#llux): %r", isread ? "read" : "write", n, addr, s->f);
			return -1;
		}
	}
	return 0;
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
	map->seg[0].rw = fdrw;
	map->seg[1].b = fp->dataddr;
	map->seg[1].e = fp->dataddr+fp->datsz;
	map->seg[1].f = fp->datoff;
	map->seg[1].fd = fd;
	map->seg[1].inuse = 1;
	map->seg[1].name = "data";
	map->seg[0].rw = fdrw;
	return map;
}
