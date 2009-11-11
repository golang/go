/*
Plan 9 from User Space src/lib9/_p9dir.c
http://code.swtch.com/plan9port/src/tip/src/lib9/_p9dir.c

Copyright 2001-2007 Russ Cox.  All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <u.h>
#define NOPLAN9DEFINES
#include <libc.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <pwd.h>
#include <grp.h>

/*
 * No need for a real disk size function here:
 * the Go build isn't looking at raw disk devices,
 * so this avoids portability problems.
 */
#define  _HAVEDISKSIZE
static vlong
disksize(int fd, int x)
{
	return 0;
}

/*
 * Caching the last group and passwd looked up is
 * a significant win (stupidly enough) on most systems.
 * It's not safe for threaded programs, but neither is using
 * getpwnam in the first place, so I'm not too worried.
 */
int
_p9dir(struct stat *lst, struct stat *st, char *name, Dir *d, char **str, char *estr)
{
	char *s;
	char tmp[20];
	static struct group *g;
	static struct passwd *p;
	static int gid, uid;
	int sz, fd;

	fd = -1;
	USED(fd);
	sz = 0;
	if(d)
		memset(d, 0, sizeof *d);

	/* name */
	s = strrchr(name, '/');
	if(s)
		s++;
	if(!s || !*s)
		s = name;
	if(*s == '/')
		s++;
	if(*s == 0)
		s = "/";
	if(d){
		if(*str + strlen(s)+1 > estr)
			d->name = "oops";
		else{
			strcpy(*str, s);
			d->name = *str;
			*str += strlen(*str)+1;
		}
	}
	sz += strlen(s)+1;

	/* user */
	if(p == nil || st->st_uid != uid || p->pw_uid != uid){
		snprint(tmp, sizeof tmp, "%d", (int)st->st_uid);
		s = tmp;
	}else
		s = p->pw_name;
	sz += strlen(s)+1;
	if(d){
		if(*str+strlen(s)+1 > estr)
			d->uid = "oops";
		else{
			strcpy(*str, s);
			d->uid = *str;
			*str += strlen(*str)+1;
		}
	}

	/* group */
	if(g == nil || st->st_gid != gid || g->gr_gid != gid){
		snprint(tmp, sizeof tmp, "%d", (int)st->st_gid);
		s = tmp;
	}else
		s = g->gr_name;
	sz += strlen(s)+1;
	if(d){
		if(*str + strlen(s)+1 > estr)
			d->gid = "oops";
		else{
			strcpy(*str, s);
			d->gid = *str;
			*str += strlen(*str)+1;
		}
	}

	if(d){
		d->type = 'M';

		d->muid = "";
		d->qid.path = ((uvlong)st->st_dev<<32) | st->st_ino;
#ifdef _HAVESTGEN
		d->qid.vers = st->st_gen;
#endif
		if(d->qid.vers == 0)
			d->qid.vers = st->st_mtime + st->st_ctime;
		d->mode = st->st_mode&0777;
		d->atime = st->st_atime;
		d->mtime = st->st_mtime;
		d->length = st->st_size;

		if(S_ISDIR(st->st_mode)){
			d->length = 0;
			d->mode |= DMDIR;
			d->qid.type = QTDIR;
		}
		if(S_ISLNK(lst->st_mode))	/* yes, lst not st */
			d->mode |= DMSYMLINK;
		if(S_ISFIFO(st->st_mode))
			d->mode |= DMNAMEDPIPE;
		if(S_ISSOCK(st->st_mode))
			d->mode |= DMSOCKET;
		if(S_ISBLK(st->st_mode)){
			d->mode |= DMDEVICE;
			d->qid.path = ('b'<<16)|st->st_rdev;
		}
		if(S_ISCHR(st->st_mode)){
			d->mode |= DMDEVICE;
			d->qid.path = ('c'<<16)|st->st_rdev;
		}
		/* fetch real size for disks */
#ifdef _HAVEDISKSIZE
		if(S_ISBLK(st->st_mode) && (fd = open(name, O_RDONLY)) >= 0){
			d->length = disksize(fd, major(st->st_dev));
			close(fd);
		}
#endif
#if defined(DIOCGMEDIASIZE)
		if(isdisk(st)){
			int fd;
			off_t mediasize;

			if((fd = open(name, O_RDONLY)) >= 0){
				if(ioctl(fd, DIOCGMEDIASIZE, &mediasize) >= 0)
					d->length = mediasize;
				close(fd);
			}
		}
#elif defined(_HAVEDISKLABEL)
		if(isdisk(st)){
			int fd, n;
			struct disklabel lab;

			if((fd = open(name, O_RDONLY)) < 0)
				goto nosize;
			if(ioctl(fd, DIOCGDINFO, &lab) < 0)
				goto nosize;
			n = minor(st->st_rdev)&7;
			if(n >= lab.d_npartitions)
				goto nosize;

			d->length = (vlong)(lab.d_partitions[n].p_size) * lab.d_secsize;

		nosize:
			if(fd >= 0)
				close(fd);
		}
#endif
	}

	return sz;
}

