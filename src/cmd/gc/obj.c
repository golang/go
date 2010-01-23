// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go.h"

/*
 * architecture-independent object file output
 */

void
dumpobj(void)
{
	bout = Bopen(outfile, OWRITE);
	if(bout == nil)
		fatal("cant open %s", outfile);

	Bprint(bout, "%s\n", thestring);
	Bprint(bout, "  exports automatically generated from\n");
	Bprint(bout, "  %s in package \"%s\"\n", curio.infile, localpkg->name);
	dumpexport();
	Bprint(bout, "\n!\n");

	outhist(bout);

	// add nil plist w AEND to catch
	// auto-generated trampolines, data
	newplist();

	dumpglobls();
	dumptypestructs();
	dumpdata();
	dumpfuncs();

	Bterm(bout);
}

void
dumpglobls(void)
{
	Node *n;
	NodeList *l;

	// add globals
	for(l=externdcl; l; l=l->next) {
		n = l->n;
		if(n->op != ONAME)
			continue;

		if(n->type == T)
			fatal("external %#N nil type\n", n);
		if(n->class == PFUNC)
			continue;
		if(n->sym->pkg != localpkg)
			continue;
		dowidth(n->type);

		// TODO(rsc): why is this not s/n->sym->def/n/ ?
		ggloblnod(n->sym->def, n->type->width);
	}
}

void
Bputname(Biobuf *b, Sym *s)
{
	Bprint(b, "%s", s->pkg->prefix);
	Bputc(b, '.');
	Bwrite(b, s->name, strlen(s->name)+1);
}

void
outhist(Biobuf *b)
{
	Hist *h;
	char *p, *q, *op;
	int n;

	for(h = hist; h != H; h = h->link) {
		p = h->name;
		op = 0;

		if(p && p[0] != '/' && h->offset == 0 && pathname && pathname[0] == '/') {
			op = p;
			p = pathname;
		}

		while(p) {
			q = utfrune(p, '/');
			if(q) {
				n = q-p;
				if(n == 0)
					n = 1;		// leading "/"
				q++;
			} else {
				n = strlen(p);
				q = 0;
			}
			if(n)
				zfile(b, p, n);
			p = q;
			if(p == 0 && op) {
				p = op;
				op = 0;
			}
		}

		zhist(b, h->line, h->offset);
	}
}

void
ieeedtod(uint64 *ieee, double native)
{
	double fr, ho, f;
	int exp;
	uint32 h, l;
	uint64 bits;

	if(native < 0) {
		ieeedtod(ieee, -native);
		*ieee |= 1ULL<<63;
		return;
	}
	if(native == 0) {
		*ieee = 0;
		return;
	}
	fr = frexp(native, &exp);
	f = 2097152L;		/* shouldnt use fp constants here */
	fr = modf(fr*f, &ho);
	h = ho;
	h &= 0xfffffL;
	f = 65536L;
	fr = modf(fr*f, &ho);
	l = ho;
	l <<= 16;
	l |= (int32)(fr*f);
	bits = ((uint64)h<<32) | l;
	if(exp < -1021) {
		// gradual underflow
		bits |= 1LL<<52;
		bits >>= -1021 - exp;
		exp = -1022;
	}
	bits |= (uint64)(exp+1022L) << 52;
	*ieee = bits;
}

int
duint8(Sym *s, int off, uint8 v)
{
	return duintxx(s, off, v, 1);
}

int
duint16(Sym *s, int off, uint16 v)
{
	return duintxx(s, off, v, 2);
}

int
duint32(Sym *s, int off, uint32 v)
{
	return duintxx(s, off, v, 4);
}

int
duint64(Sym *s, int off, uint64 v)
{
	return duintxx(s, off, v, 8);
}

int
duintptr(Sym *s, int off, uint64 v)
{
	return duintxx(s, off, v, widthptr);
}
