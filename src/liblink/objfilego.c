// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Writing of internal program representation to a serialized form
// so that the Go translation of these routines can do the actual
// program layout.
// The serialized form and this code support the piecewise transition
// from C to Go and will be removed along with the rest of the C code
// when it is no longer needed.
// There has been no attempt to make it particularly efficient, nor will there be.

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <link.h>

/*c2go

char *mktempdir(void);
int runcmd(char**);
void removeall(char*);
*/

static void printtype(Link*, Biobuf*, int);
static void printsym(Link*, Biobuf*, LSym*);
static void printprog(Link*, Biobuf*, Prog*);
static void printaddr(Link*, Biobuf*, Addr*);
static void printhist(Link*, Biobuf*, Hist*);
static void printint(Link*, Biobuf*, int64);
static void printstr(Link*, Biobuf*, char*);
static void printptr(Link*, Biobuf*, void*);

#undef waitpid

enum
{
	TypeEnd = 0,
	TypeCtxt,
	TypePlist,
	TypeSym,
	TypeProg,
	TypeAddr,
	TypeHist,
};

void
writeobjgo1(Link *ctxt, char *outfile)
{
	int i;
	char *p;
	Biobuf *bw;
	Plist *pl;
	
	p = smprint("%s.goliblink.in", outfile);
	bw = Bopen(p, OWRITE);
	if(bw == nil)
		sysfatal("writing liblinktest input: %r");

	printtype(ctxt, bw, TypeCtxt);
	printstr(ctxt, bw, ctxt->arch->name);
	printint(ctxt, bw, ctxt->goarm);
	printint(ctxt, bw, ctxt->debugasm);
	printstr(ctxt, bw, ctxt->trimpath);
	printptr(ctxt, bw, ctxt->plist);
	printptr(ctxt, bw, ctxt->plast);
	printptr(ctxt, bw, ctxt->hist);
	printptr(ctxt, bw, ctxt->ehist);
	for(i = 0; i < LINKHASH; i++) {
		if(ctxt->hash[i] != nil) {
			printint(ctxt, bw, i);
			printptr(ctxt, bw, ctxt->hash[i]);
		}
	}
	printint(ctxt, bw, -1);

	printhist(ctxt, bw, ctxt->hist);
	printhist(ctxt, bw, ctxt->ehist);

	for(pl=ctxt->plist; pl != nil; pl = pl->link) {
		printtype(ctxt, bw, TypePlist);
		printptr(ctxt, bw, pl);
		printint(ctxt, bw, pl->recur);
		printptr(ctxt, bw, pl->name);
		printptr(ctxt, bw, pl->firstpc);
		printptr(ctxt, bw, pl->link);
		printsym(ctxt, bw, pl->name);
		printprog(ctxt, bw, pl->firstpc);
	}
	
	for(i = 0; i < LINKHASH; i++)
		printsym(ctxt, bw, ctxt->hash[i]);

	printtype(ctxt, bw, TypeEnd);
	Bterm(bw);
}

void
writeobjgo2(Link *ctxt, char *outfile, int64 offset)
{
	char *p, *env, *prog, *cmd[10];
	char offsetbuf[20];
	
	USED(ctxt);

	env = getenv("GOOBJWRITER");
	if(env != nil && env[0] != '\0')
		prog = env;
	else
		prog = smprint("%s/pkg/tool/%s_%s/objwriter", getgoroot(), getgohostos(), getgohostarch());

	p = smprint("%s.goliblink.in", outfile);
	
	snprint(offsetbuf, sizeof offsetbuf, "%lld", offset);
	
	cmd[0] = prog;
	cmd[1] = p;
	cmd[2] = outfile;
	cmd[3] = offsetbuf;
	cmd[4] = ctxt->arch->name;
	cmd[5] = nil;
	if(runcmd(cmd) < 0)
		sysfatal("running %s: %r", prog);

	env = getenv("GOOBJ");
	if(env == nil || atoi(env) <= 2)
		remove(p);
}

static void
printtype(Link *ctxt, Biobuf *bw, int t)
{
	printint(ctxt, bw, t);
}

static void
printint(Link *ctxt, Biobuf *bw, int64 v)
{
	uint64 u;
	
	USED(ctxt);

	u = (uint64)(v<<1) ^ (uint64)(v>>63);
	while(u >= 0x80) {
		Bputc(bw, u&0x7F | 0x80);
		u >>= 7;
	}
	Bputc(bw, u);
}

static void
printstr(Link *ctxt, Biobuf *bw, char *s)
{
	if(s == nil)
		s = "";
	printint(ctxt, bw, strlen(s));
	Bwrite(bw, s, strlen(s));
}

static void
printptr(Link *ctxt, Biobuf *bw, void *v)
{
	printint(ctxt, bw, (int64)(uintptr)v);
}

static void
printsym(Link *ctxt, Biobuf *bw, LSym *s)
{
	int i;
	Reloc *r;

	if(s == nil || s->printed)
		return;
	s->printed = 1;
	printtype(ctxt, bw, TypeSym);
	printptr(ctxt, bw, s);
	printstr(ctxt, bw, s->name);
	printstr(ctxt, bw, s->extname);
	printint(ctxt, bw, s->type);
	printint(ctxt, bw, s->version);
	printint(ctxt, bw, s->dupok);
	printint(ctxt, bw, s->external);
	printint(ctxt, bw, s->nosplit);
	printint(ctxt, bw, s->reachable);
	printint(ctxt, bw, s->cgoexport);
	printint(ctxt, bw, s->special);
	printint(ctxt, bw, s->stkcheck);
	printint(ctxt, bw, s->hide);
	printint(ctxt, bw, s->leaf);
	printint(ctxt, bw, s->fnptr);
	printint(ctxt, bw, s->seenglobl);
	printint(ctxt, bw, s->onlist);
	printint(ctxt, bw, s->symid);
	printint(ctxt, bw, s->dynid);
	printint(ctxt, bw, s->sig);
	printint(ctxt, bw, s->plt);
	printint(ctxt, bw, s->got);
	printint(ctxt, bw, s->align);
	printint(ctxt, bw, s->elfsym);
	printint(ctxt, bw, s->args);
	printint(ctxt, bw, s->locals);
	printint(ctxt, bw, s->value);
	printint(ctxt, bw, s->size);
	printptr(ctxt, bw, s->hash);
	printptr(ctxt, bw, s->allsym);
	printptr(ctxt, bw, s->next);
	printptr(ctxt, bw, s->sub);
	printptr(ctxt, bw, s->outer);
	printptr(ctxt, bw, s->gotype);
	printptr(ctxt, bw, s->reachparent);
	printptr(ctxt, bw, s->queue);
	printstr(ctxt, bw, s->file);
	printstr(ctxt, bw, s->dynimplib);
	printstr(ctxt, bw, s->dynimpvers);
	printptr(ctxt, bw, s->text);
	printptr(ctxt, bw, s->etext);
	printint(ctxt, bw, s->np);
	Bwrite(bw, s->p, s->np);
	printint(ctxt, bw, s->nr);
	for(i=0; i<s->nr; i++) {
		r = s->r+i;
		printint(ctxt, bw, r->off);
		printint(ctxt, bw, r->siz);
		printint(ctxt, bw, r->done);
		printint(ctxt, bw, r->type);
		printint(ctxt, bw, r->add);
		printint(ctxt, bw, r->xadd);
		printptr(ctxt, bw, r->sym);
		printptr(ctxt, bw, r->xsym);
	}
	
	printsym(ctxt, bw, s->hash);
	printsym(ctxt, bw, s->allsym);
	printsym(ctxt, bw, s->next);
	printsym(ctxt, bw, s->sub);
	printsym(ctxt, bw, s->outer);
	printsym(ctxt, bw, s->gotype);
	printsym(ctxt, bw, s->reachparent);
	printsym(ctxt, bw, s->queue);
	printprog(ctxt, bw, s->text);
	printprog(ctxt, bw, s->etext);
	for(i=0; i<s->nr; i++) {
		r = s->r+i;
		printsym(ctxt, bw, r->sym);
		printsym(ctxt, bw, r->xsym);
	}
}

static void
printprog(Link *ctxt, Biobuf *bw, Prog *p0)
{
	Prog *p, *q;

	for(p = p0; p != nil && !p->printed; p=p->link) {
		p->printed = 1;
	
		printtype(ctxt, bw, TypeProg);
		printptr(ctxt, bw, p);
		printint(ctxt, bw, p->pc);
		printint(ctxt, bw, p->lineno);
		printptr(ctxt, bw, p->link);
		printint(ctxt, bw, p->as);
		printint(ctxt, bw, p->reg);
		printint(ctxt, bw, p->scond);
		printint(ctxt, bw, p->width);
		printaddr(ctxt, bw, &p->from);
		printaddr(ctxt, bw, &p->from3);
		printaddr(ctxt, bw, &p->to);
		printsym(ctxt, bw, p->from.sym);
		printsym(ctxt, bw, p->from.gotype);
		printsym(ctxt, bw, p->to.sym);
		printsym(ctxt, bw, p->to.gotype);
	}
	
	q = p;
	for(p=p0; p!=q; p=p->link) {
		if(p->from.type == TYPE_BRANCH)
			printprog(ctxt, bw, p->from.u.branch);
		if(p->to.type == TYPE_BRANCH)
			printprog(ctxt, bw, p->to.u.branch);
	}
}

static void
printaddr(Link *ctxt, Biobuf *bw, Addr *a)
{
	static char zero[8];

	printtype(ctxt, bw, TypeAddr);
	printint(ctxt, bw, a->offset);
	if(a->type == TYPE_FCONST) {
		uint64 u;
		float64 f;
		f = a->u.dval;
		memmove(&u, &f, 8);
		printint(ctxt, bw, u);
	} else
		printint(ctxt, bw, 0);
	if(a->type == TYPE_SCONST)
		Bwrite(bw, a->u.sval, 8);
	else
		Bwrite(bw, zero, 8);
	if(a->type == TYPE_BRANCH)
		printptr(ctxt, bw, a->u.branch);
	else	
		printptr(ctxt, bw, nil);
	printptr(ctxt, bw, a->sym);
	printptr(ctxt, bw, a->gotype);
	printint(ctxt, bw, a->type);
	printint(ctxt, bw, a->index);
	printint(ctxt, bw, a->scale);
	printint(ctxt, bw, a->reg);
	printint(ctxt, bw, a->name);
	printint(ctxt, bw, a->class);
	printint(ctxt, bw, a->etype);
	if(a->type == TYPE_TEXTSIZE)
		printint(ctxt, bw, a->u.argsize);
	else
		printint(ctxt, bw, 0);
	printint(ctxt, bw, a->width);
}

static void
printhist(Link *ctxt, Biobuf *bw, Hist *h)
{
	if(h == nil || h->printed)
		return;
	h->printed = 1;

	printtype(ctxt, bw, TypeHist);
	printptr(ctxt, bw, h);
	printptr(ctxt, bw, h->link);
	if(h->name == nil)
		printstr(ctxt, bw, "<pop>");
	else
		printstr(ctxt, bw, h->name);
	printint(ctxt, bw, h->line);
	printint(ctxt, bw, h->offset);
	printhist(ctxt, bw, h->link);
}
