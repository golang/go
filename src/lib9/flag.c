// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>

// Flag hash.
typedef struct Flag Flag;

struct Flag
{
	char *name;
	int namelen;
	char *desc;
	int iscount;
	void (*set)(char*, void*);
	void (*set2)(char*, char*, void*);
	void *arg;
	Flag *next;
	Flag *allnext;
};

static Flag *curflag;

static Flag *fhash[512];
static Flag *first, *last;

char *argv0;

/*
 * Mac OS can't deal with files that only declare data.
 * ARGBEGIN mentions this function so that this file gets pulled in.
 */
void __fixargv0(void) { }

// FNV-1 hash. http://isthe.com/chongo/tech/comp/fnv/
static uint32
fnv(char *p, int n)
{
	uint32 h;
	
	h = 2166136261U;
	while(n-- > 0)
		h = (h*16777619) ^ (uchar)*p++;
	return h;
}

static Flag*
lookflag(char *name, int namelen, int creat)
{
	uint32 h;
	Flag *f;

	h = fnv(name, namelen) & (nelem(fhash)-1);
	for(f=fhash[h]; f; f=f->next) {
		if(f->namelen == namelen && memcmp(f->name, name, (size_t)namelen) == 0) {
			if(creat)
				sysfatal("multiple definitions of flag -%s", name);
			return f;
		}
	}
	
	if(!creat)
		return nil;

	f = malloc(sizeof *f);
	if(f == nil)
		sysfatal("out of memory");
	memset(f, 0, sizeof *f);
	f->name = name;
	f->namelen = namelen;
	f->next = fhash[h];
	if(first == nil)
		first = f;
	else
		last->allnext = f;
	last = f;
	fhash[h] = f;
	return f;
}

static void
count(char *arg, void *p)
{
	int *ip;
	
	ip = p;
	if(arg != nil)
		*ip = atoi(arg);
	else
		(*ip)++;
}

void
flagcount(char *name, char *desc, int *p)
{
	Flag *f;
	
	f = lookflag(name, (int)strlen(name), 1);
	f->desc = desc;
	f->iscount = 1;
	f->set = count;
	f->arg = p;
}

static void
atollwhex(char *s, void *p)
{
	char *t;

	*(int64*)p = strtoll(s, &t, 0);
	if(*s == '\0' || *t != '\0')
		sysfatal("invalid numeric argument -%s=%s", curflag->name, s);
}

void
flagint64(char *name, char *desc, int64 *p)
{
	Flag *f;
	
	f = lookflag(name, (int)strlen(name), 1);
	f->desc = desc;
	f->set = atollwhex;
	f->arg = p;
}

static void
atolwhex(char *s, void *p)
{
	char *t;

	*(int32*)p = (int32)strtol(s, &t, 0);
	if(*s == '\0' || *t != '\0')
		sysfatal("invalid numeric argument -%s=%s", curflag->name, s);
}

void
flagint32(char *name, char *desc, int32 *p)
{
	Flag *f;
	
	f = lookflag(name, (int)strlen(name), 1);
	f->desc = desc;
	f->set = atolwhex;
	f->arg = p;
}

static void
string(char *s, void *p)
{
	*(char**)p = s;
}

void
flagstr(char *name, char *desc, char **p)
{

	Flag *f;
	
	f = lookflag(name, (int)strlen(name), 1);
	f->desc = desc;
	f->set = string;
	f->arg = p;
}	

static void
fn0(char *s, void *p)
{
	USED(s);
	((void(*)(void))p)();
}

void
flagfn0(char *name, char *desc, void (*fn)(void))
{
	Flag *f;
	
	f = lookflag(name, (int)strlen(name), 1);
	f->desc = desc;
	f->set = fn0;
	f->arg = fn;
	f->iscount = 1;
}

static void
fn1(char *s, void *p)
{
	((void(*)(char*))p)(s);
}

void
flagfn1(char *name, char *desc, void (*fn)(char*))
{
	Flag *f;
	
	f = lookflag(name, (int)strlen(name), 1);
	f->desc = desc;
	f->set = fn1;
	f->arg = fn;
}

static void
fn2(char *s, char *t, void *p)
{
	((void(*)(char*, char*))p)(s, t);
}

void
flagfn2(char *name, char *desc, void (*fn)(char*, char*))
{
	Flag *f;
	
	f = lookflag(name, (int)strlen(name), 1);
	f->desc = desc;
	f->set2 = fn2;
	f->arg = fn;
}

void
flagparse(int *argcp, char ***argvp, void (*usage)(void))
{
	int argc;
	char **argv, *p, *q;
	char *name;
	int namelen;
	Flag *f;
	
	argc = *argcp;
	argv = *argvp;

	argv0 = argv[0];
	argc--;
	argv++;
	
	while(argc > 0) {
		p = *argv;
		// stop before non-flag or -
		if(*p != '-' || p[1] == '\0')
			break;
		argc--;
		argv++;
		// stop after --
		if(p[1] == '-' && p[2] == '\0') {
			break;
		}
		
		// turn --foo into -foo
		if(p[1] == '-' && p[2] != '-')
			p++;
		
		// allow -flag=arg if present
		name = p+1;
		q = strchr(name, '=');
		if(q != nil)
			namelen = (int)(q++ - name);
		else
			namelen = (int)strlen(name);
		f = lookflag(name, namelen, 0);
		if(f == nil) {
			if(strcmp(p, "-h") == 0 || strcmp(p, "-help") == 0 || strcmp(p, "-?") == 0)
				usage();
			sysfatal("unknown flag %s", p);
		}
		curflag = f;

		// otherwise consume next argument if non-boolean
		if(!f->iscount && q == nil) {
			if(argc-- == 0)
				sysfatal("missing argument to flag %s", p);
			q = *argv++;
		}
		
		// and another if we need two
		if(f->set2 != nil) {
			if(argc-- == 0)
				sysfatal("missing second argument to flag %s", p);
			f->set2(q, *argv++, f->arg);
			continue;
		}

		f->set(q, f->arg);			
	}
	
	*argcp = argc;
	*argvp = argv;		
}

void
flagprint(int fd)
{
	Flag *f;
	char *p, *q;
	
	for(f=first; f; f=f->allnext) {
		p = f->desc;
		if(p == nil || *p == '\0') // undocumented flag
			continue;
		q = strstr(p, ": ");
		if(q)
			fprint(fd, "  -%s %.*s\n    \t%s\n", f->name, utfnlen(p, q-p), p, q+2);
		else if(f->namelen > 1)
			fprint(fd, "  -%s\n    \t%s\n", f->name, p);
		else
			fprint(fd, "  -%s\t%s\n", f->name, p);
	}
}
