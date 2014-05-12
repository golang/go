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

// Does s have t as a path prefix?
// That is, does s == t or does s begin with t followed by a slash?
// For portability, we allow ASCII case folding, so that haspathprefix("a/b/c", "A/B") is true.
// Similarly, we allow slash folding, so that haspathprefix("a/b/c", "a\\b") is true.
static int
haspathprefix(char *s, char *t)
{
	int i, cs, ct;

	if(t == nil)
		return 0;
	for(i=0; t[i]; i++) {
		cs = s[i];
		ct = t[i];
		if('A' <= cs && cs <= 'Z')
			cs += 'a' - 'A';
		if('A' <= ct && ct <= 'Z')
			ct += 'a' - 'A';
		if(cs == '\\')
			cs = '/';
		if(ct == '\\')
			ct = '/';
		if(cs != ct)
			return 0;
	}
	return s[i] == '\0' || s[i] == '/' || s[i] == '\\';
}

// This is a simplified copy of linklinefmt above.
// It doesn't allow printing the full stack, and it returns the file name and line number separately.
// TODO: Unify with linklinefmt somehow.
void
linkgetline(Link *ctxt, int32 line, LSym **f, int32 *l)
{
	struct
	{
		Hist*	incl;	/* start of this include file */
		int32	idel;	/* delta line number to apply to include */
		Hist*	line;	/* start of this #line directive */
		int32	ldel;	/* delta line number to apply to #line */
	} a[HISTSZ];
	int32 lno, d, dlno;
	int n;
	Hist *h;
	char buf[1024], buf1[1024], *file;

	lno = line;
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

	if(n <= 0) {
		*f = linklookup(ctxt, "??", HistVersion);
		*l = 0;
		return;
	}
	
	n--;
	if(a[n].line) {
		file = a[n].line->name;
		dlno = a[n].ldel-1;
	} else {
		file = a[n].incl->name;
		dlno = a[n].idel-1;
	}
	if((!ctxt->windows && file[0] == '/') || (ctxt->windows && file[1] == ':') || file[0] == '<')
		snprint(buf, sizeof buf, "%s", file);
	else
		snprint(buf, sizeof buf, "%s/%s", ctxt->pathname, file);

	// Remove leading ctxt->trimpath, or else rewrite $GOROOT to $GOROOT_FINAL.
	if(haspathprefix(buf, ctxt->trimpath)) {
		if(strlen(buf) == strlen(ctxt->trimpath))
			strcpy(buf, "??");
		else {
			snprint(buf1, sizeof buf1, "%s", buf+strlen(ctxt->trimpath)+1);
			if(buf1[0] == '\0')
				strcpy(buf1, "??");
			strcpy(buf, buf1);
		}
	} else if(ctxt->goroot_final != nil && haspathprefix(buf, ctxt->goroot)) {
		snprint(buf1, sizeof buf1, "%s%s", ctxt->goroot_final, buf+strlen(ctxt->goroot));
		strcpy(buf, buf1);
	}

	lno -= dlno;
	*f = linklookup(ctxt, buf, HistVersion);
	*l = lno;
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
