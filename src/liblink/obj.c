// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <link.h>

enum
{
	HISTSZ = 10,
	NSYM = 50,
};

int
linklinefmt(Link *ctxt, Fmt *fp)
{
	struct
	{
		Hist*	incl;	/* start of this include file */
		int32	idel;	/* delta line number to apply to include */
		Hist*	line;	/* start of this #line directive */
		int32	ldel;	/* delta line number to apply to #line */
	} a[HISTSZ];
	int32 lno, d;
	int i, n;
	Hist *h;

	lno = va_arg(fp->args, int32);

	n = 0;
	for(h=ctxt->hist; h!=nil; h=h->link) {
		if(h->offset < 0)
			continue;
		if(lno < h->line)
			break;
		if(h->name) {
			if(h->offset > 0) {
				// #line directive
				if(n > 0 && n < HISTSZ) {
					a[n-1].line = h;
					a[n-1].ldel = h->line - h->offset + 1;
				}
			} else {
				// beginning of file
				if(n < HISTSZ) {
					a[n].incl = h;
					a[n].idel = h->line;
					a[n].line = 0;
				}
				n++;
			}
			continue;
		}
		n--;
		if(n > 0 && n < HISTSZ) {
			d = h->line - a[n].incl->line;
			a[n-1].ldel += d;
			a[n-1].idel += d;
		}
	}

	if(n > HISTSZ)
		n = HISTSZ;

	for(i=n-1; i>=0; i--) {
		if(i != n-1) {
			if(fp->flags & ~(FmtWidth|FmtPrec))
				break;
			fmtprint(fp, " ");
		}
		if(ctxt->debugline || (fp->flags&FmtLong))
			fmtprint(fp, "%s/", ctxt->pathname);
		if(a[i].line)
			fmtprint(fp, "%s:%d[%s:%d]",
				a[i].line->name, lno-a[i].ldel+1,
				a[i].incl->name, lno-a[i].idel+1);
		else
			fmtprint(fp, "%s:%d",
				a[i].incl->name, lno-a[i].idel+1);
		lno = a[i].incl->line - 1;	// now print out start of this file
	}
	if(n == 0)
		fmtprint(fp, "<unknown line number>");

	return 0;
}

static void
outzfile(Link *ctxt, Biobuf *b, char *p)
{
	char *q, *q2;

	while(p) {
		q = utfrune(p, '/');
		if(ctxt->windows) {
			q2 = utfrune(p, '\\');
			if(q2 && (!q || q2 < q))
				q = q2;
		}
		if(!q) {
			ctxt->arch->zfile(b, p, strlen(p));
			return;
		}
		if(q > p)
			ctxt->arch->zfile(b, p, q-p);
		p = q + 1;
	}
}

#define isdelim(c) (c == '/' || c == '\\')

static void
outwinname(Link *ctxt, Biobuf *b, Hist *h, char *ds, char *p)
{
	if(isdelim(p[0])) {
		// full rooted name
		ctxt->arch->zfile(b, ds, 3);	// leading "c:/"
		outzfile(ctxt, b, p+1);
	} else {
		// relative name
		if(h->offset >= 0 && ctxt->pathname && ctxt->pathname[1] == ':') {
			if(tolowerrune(ds[0]) == tolowerrune(ctxt->pathname[0])) {
				// using current drive
				ctxt->arch->zfile(b, ctxt->pathname, 3);	// leading "c:/"
				outzfile(ctxt, b, ctxt->pathname+3);
			} else {
				// using drive other then current,
				// we don't have any simple way to
				// determine current working directory
				// there, therefore will output name as is
				ctxt->arch->zfile(b, ds, 2);	// leading "c:"
			}
		}
		outzfile(ctxt, b, p);
	}
}

void
linkouthist(Link *ctxt, Biobuf *b)
{
	Hist *h;
	char *p, ds[] = {'c', ':', '/', 0};
	char *tofree;
	int n;
	static int first = 1;
	static char *goroot, *goroot_final;

	if(first) {
		// Decide whether we need to rewrite paths from $GOROOT to $GOROOT_FINAL.
		first = 0;
		goroot = getenv("GOROOT");
		goroot_final = getenv("GOROOT_FINAL");
		if(goroot == nil)
			goroot = "";
		if(goroot_final == nil)
			goroot_final = goroot;
		if(strcmp(goroot, goroot_final) == 0) {
			goroot = nil;
			goroot_final = nil;
		}
	}

	tofree = nil;
	for(h = ctxt->hist; h != nil; h = h->link) {
		p = h->name;
		if(p) {
			if(goroot != nil) {
				n = strlen(goroot);
				if(strncmp(p, goroot, strlen(goroot)) == 0 && p[n] == '/') {
					tofree = smprint("%s%s", goroot_final, p+n);
					p = tofree;
				}
			}
			if(ctxt->windows) {
				// if windows variable is set, then, we know already,
				// pathname is started with windows drive specifier
				// and all '\' were replaced with '/' (see lex.c)
				if(isdelim(p[0]) && isdelim(p[1])) {
					// file name has network name in it, 
					// like \\server\share\dir\file.go
					ctxt->arch->zfile(b, "//", 2);	// leading "//"
					outzfile(ctxt, b, p+2);
				} else if(p[1] == ':') {
					// file name has drive letter in it
					ds[0] = p[0];
					outwinname(ctxt, b, h, ds, p+2);
				} else {
					// no drive letter in file name
					outwinname(ctxt, b, h, ctxt->pathname, p);
				}
			} else {
				if(p[0] == '/') {
					// full rooted name, like /home/rsc/dir/file.go
					ctxt->arch->zfile(b, "/", 1);	// leading "/"
					outzfile(ctxt, b, p+1);
				} else {
					// relative name, like dir/file.go
					if(h->offset >= 0 && ctxt->pathname && ctxt->pathname[0] == '/') {
						ctxt->arch->zfile(b, "/", 1);	// leading "/"
						outzfile(ctxt, b, ctxt->pathname+1);
					}
					outzfile(ctxt, b, p);
				}
			}
		}
		ctxt->arch->zhist(b, h->line, h->offset);
		if(tofree) {
			free(tofree);
			tofree = nil;
		}
	}
}

void
linklinehist(Link *ctxt, int lineno, char *f, int offset)
{
	Hist *h;

	if(0) // debug['f']
		if(f) {
			if(offset)
				print("%4d: %s (#line %d)\n", lineno, f, offset);
			else
				print("%4d: %s\n", lineno, f);
		} else
			print("%4d: <pop>\n", lineno);

	h = malloc(sizeof(Hist));
	memset(h, 0, sizeof *h);
	h->name = f;
	h->line = lineno;
	h->offset = offset;
	h->link = nil;
	if(ctxt->ehist == nil) {
		ctxt->hist = h;
		ctxt->ehist = h;
		return;
	}
	ctxt->ehist->link = h;
	ctxt->ehist = h;
}

void
linkprfile(Link *ctxt, int32 l)
{
	int i, n;
	Hist a[HISTSZ], *h;
	int32 d;

	n = 0;
	for(h = ctxt->hist; h != nil; h = h->link) {
		if(l < h->line)
			break;
		if(h->name) {
			if(h->offset == 0) {
				if(n >= 0 && n < HISTSZ)
					a[n] = *h;
				n++;
				continue;
			}
			if(n > 0 && n < HISTSZ)
				if(a[n-1].offset == 0) {
					a[n] = *h;
					n++;
				} else
					a[n-1] = *h;
			continue;
		}
		n--;
		if(n >= 0 && n < HISTSZ) {
			d = h->line - a[n].line;
			for(i=0; i<n; i++)
				a[i].line += d;
		}
	}
	if(n > HISTSZ)
		n = HISTSZ;
	for(i=0; i<n; i++)
		print("%s:%ld ", a[i].name, (long)(l-a[i].line+a[i].offset+1));
}

/*
 * start a new Prog list.
 */
Plist*
linknewplist(Link *ctxt)
{
	Plist *pl;

	pl = malloc(sizeof(*pl));
	memset(pl, 0, sizeof *pl);
	if(ctxt->plist == nil)
		ctxt->plist = pl;
	else
		ctxt->plast->link = pl;
	ctxt->plast = pl;

	return pl;
}

static struct {
	struct { LSym *sym; short type; } h[NSYM];
	int sym;
} z;

static void
zsymreset(void)
{
	for(z.sym=0; z.sym<NSYM; z.sym++) {
		z.h[z.sym].sym = nil;
		z.h[z.sym].type = 0;
	}
	z.sym = 1;
}


static int
zsym(Link *ctxt, Biobuf *b, LSym *s, int t, int *new)
{
	int i;

	*new = 0;
	if(s == nil)
		return 0;

	i = s->symid;
	if(i < 0 || i >= NSYM)
		i = 0;
	if(z.h[i].type == t && z.h[i].sym == s)
		return i;
	i = z.sym;
	s->symid = i;
	ctxt->arch->zname(b, s, t);
	z.h[i].sym = s;
	z.h[i].type = t;
	if(++z.sym >= NSYM)
		z.sym = 1;
	*new = 1;
	return i;
}

static int
zsymaddr(Link *ctxt, Biobuf *b, Addr *a, int *new)
{
	return zsym(ctxt, b, a->sym, ctxt->arch->symtype(a), new);
}

void
linkwritefuncs(Link *ctxt, Biobuf *b)
{
	int32 pcloc;
	Plist *pl;
	LSym *s;
	Prog *p;
	int sf, st, gf, gt, new;

	zsymreset();

	// fix up pc
	pcloc = 0;
	for(pl=ctxt->plist; pl!=nil; pl=pl->link) {
		if(pl->name != nil && strcmp(pl->name->name, "_") == 0)
			continue;
		for(p=pl->firstpc; p!=nil; p=p->link) {
			p->loc = pcloc;
			if(!ctxt->arch->isdata(p))
				pcloc++;
		}
	}

	// put out functions
	for(pl=ctxt->plist; pl!=nil; pl=pl->link) {
		if(pl->name != nil && strcmp(pl->name->name, "_") == 0)
			continue;

		// -S prints code; -S -S prints code and data
		if(ctxt->debugasm && (pl->name || ctxt->debugasm>1)) {
			s = pl->name;
			print("\n--- prog list \"%lS\" ---\n", s);
			for(p=pl->firstpc; p!=nil; p=p->link)
				print("%P\n", p);
		}

		for(p=pl->firstpc; p!=nil; p=p->link) {
			for(;;) {
				sf = zsymaddr(ctxt, b, &p->from, &new);
				gf = zsym(ctxt, b, p->from.gotype, ctxt->arch->D_EXTERN, &new);
				if(new && sf == gf)
					continue;
				st = zsymaddr(ctxt, b, &p->to, &new);
				if(new && (st == sf || st == gf))
					continue;
				gt = zsym(ctxt, b, p->to.gotype, ctxt->arch->D_EXTERN, &new);
				if(new && (gt == sf || gt == gf || gt == st))
					continue;
				break;
			}
			ctxt->arch->zprog(ctxt, b, p, sf, gf, st, gt);
		}
	}
}
