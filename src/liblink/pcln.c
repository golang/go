// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <link.h>

static void
addvarint(Link *ctxt, Pcdata *d, uint32 val)
{
	int32 n;
	uint32 v;
	uchar *p;

	USED(ctxt);

	n = 0;
	for(v = val; v >= 0x80; v >>= 7)
		n++;
	n++;

	if(d->n + n > d->m) {
		d->m = (d->n + n)*2;
		d->p = erealloc(d->p, d->m);
	}

	p = d->p + d->n;
	for(v = val; v >= 0x80; v >>= 7)
		*p++ = v | 0x80;
	*p = v;
	d->n += n;
}

// funcpctab writes to dst a pc-value table mapping the code in func to the values
// returned by valfunc parameterized by arg. The invocation of valfunc to update the
// current value is, for each p,
//
//	val = valfunc(func, val, p, 0, arg);
//	record val as value at p->pc;
//	val = valfunc(func, val, p, 1, arg);
//
// where func is the function, val is the current value, p is the instruction being
// considered, and arg can be used to further parameterize valfunc.
static void
funcpctab(Link *ctxt, Pcdata *dst, LSym *func, char *desc, int32 (*valfunc)(Link*, LSym*, int32, Prog*, int32, void*), void* arg)
{
	int dbg, i;
	int32 oldval, val, started;
	uint32 delta;
	vlong pc;
	Prog *p;

	// To debug a specific function, uncomment second line and change name.
	dbg = 0;
	//dbg = strcmp(func->name, "main.main") == 0;
	//dbg = strcmp(desc, "pctofile") == 0;

	ctxt->debugpcln += dbg;

	dst->n = 0;

	if(ctxt->debugpcln)
		Bprint(ctxt->bso, "funcpctab %s [valfunc=%s]\n", func->name, desc);

	val = -1;
	oldval = val;
	if(func->text == nil) {
		ctxt->debugpcln -= dbg;
		return;
	}

	pc = func->text->pc;
	
	if(ctxt->debugpcln)
		Bprint(ctxt->bso, "%6llux %6d %P\n", pc, val, func->text);

	started = 0;
	for(p=func->text; p != nil; p = p->link) {
		// Update val. If it's not changing, keep going.
		val = valfunc(ctxt, func, val, p, 0, arg);
		if(val == oldval && started) {
			val = valfunc(ctxt, func, val, p, 1, arg);
			if(ctxt->debugpcln)
				Bprint(ctxt->bso, "%6llux %6s %P\n", (vlong)p->pc, "", p);
			continue;
		}

		// If the pc of the next instruction is the same as the
		// pc of this instruction, this instruction is not a real
		// instruction. Keep going, so that we only emit a delta
		// for a true instruction boundary in the program.
		if(p->link && p->link->pc == p->pc) {
			val = valfunc(ctxt, func, val, p, 1, arg);
			if(ctxt->debugpcln)
				Bprint(ctxt->bso, "%6llux %6s %P\n", (vlong)p->pc, "", p);
			continue;
		}

		// The table is a sequence of (value, pc) pairs, where each
		// pair states that the given value is in effect from the current position
		// up to the given pc, which becomes the new current position.
		// To generate the table as we scan over the program instructions,
		// we emit a "(value" when pc == func->value, and then
		// each time we observe a change in value we emit ", pc) (value".
		// When the scan is over, we emit the closing ", pc)".
		//
		// The table is delta-encoded. The value deltas are signed and
		// transmitted in zig-zag form, where a complement bit is placed in bit 0,
		// and the pc deltas are unsigned. Both kinds of deltas are sent
		// as variable-length little-endian base-128 integers,
		// where the 0x80 bit indicates that the integer continues.

		if(ctxt->debugpcln)
			Bprint(ctxt->bso, "%6llux %6d %P\n", (vlong)p->pc, val, p);

		if(started) {
			addvarint(ctxt, dst, (p->pc - pc) / ctxt->arch->minlc);
			pc = p->pc;
		}
		delta = val - oldval;
		if(delta>>31)
			delta = 1 | ~(delta<<1);
		else
			delta <<= 1;
		addvarint(ctxt, dst, delta);
		oldval = val;
		started = 1;
		val = valfunc(ctxt, func, val, p, 1, arg);
	}

	if(started) {
		if(ctxt->debugpcln)
			Bprint(ctxt->bso, "%6llux done\n", (vlong)func->text->pc+func->size);
		addvarint(ctxt, dst, (func->value+func->size - pc) / ctxt->arch->minlc);
		addvarint(ctxt, dst, 0); // terminator
	}

	if(ctxt->debugpcln) {
		Bprint(ctxt->bso, "wrote %d bytes to %p\n", dst->n, dst);
		for(i=0; i<dst->n; i++)
			Bprint(ctxt->bso, " %02ux", dst->p[i]);
		Bprint(ctxt->bso, "\n");
	}

	ctxt->debugpcln -= dbg;
}

// pctofileline computes either the file number (arg == 0)
// or the line number (arg == 1) to use at p.
// Because p->lineno applies to p, phase == 0 (before p)
// takes care of the update.
static int32
pctofileline(Link *ctxt, LSym *sym, int32 oldval, Prog *p, int32 phase, void *arg)
{
	int32 i, l;
	LSym *f;
	Pcln *pcln;

	USED(sym);

	if(p->as == ctxt->arch->ATEXT || p->as == ctxt->arch->ANOP || p->as == ctxt->arch->AUSEFIELD || p->lineno == 0 || phase == 1)
		return oldval;
	linkgetline(ctxt, p->lineno, &f, &l);
	if(f == nil) {
	//	print("getline failed for %s %P\n", ctxt->cursym->name, p);
		return oldval;
	}
	if(arg == nil)
		return l;
	pcln = arg;
	
	if(f == pcln->lastfile)
		return pcln->lastindex;

	for(i=0; i<pcln->nfile; i++) {
		if(pcln->file[i] == f) {
			pcln->lastfile = f;
			pcln->lastindex = i;
			return i;
		}
	}

	if(pcln->nfile >= pcln->mfile) {
		pcln->mfile = (pcln->nfile+1)*2;
		pcln->file = erealloc(pcln->file, pcln->mfile*sizeof pcln->file[0]);
	}
	pcln->file[pcln->nfile++] = f;
	pcln->lastfile = f;
	pcln->lastindex = i;
	return i;
}

// pctospadj computes the sp adjustment in effect.
// It is oldval plus any adjustment made by p itself.
// The adjustment by p takes effect only after p, so we
// apply the change during phase == 1.
static int32
pctospadj(Link *ctxt, LSym *sym, int32 oldval, Prog *p, int32 phase, void *arg)
{
	USED(arg);
	USED(sym);

	if(oldval == -1) // starting
		oldval = 0;
	if(phase == 0)
		return oldval;
	if(oldval + p->spadj < -10000 || oldval + p->spadj > 1100000000) {
		ctxt->diag("overflow in spadj: %d + %d = %d", oldval, p->spadj, oldval + p->spadj);
		sysfatal("bad code");
	}
	return oldval + p->spadj;
}

// pctopcdata computes the pcdata value in effect at p.
// A PCDATA instruction sets the value in effect at future
// non-PCDATA instructions.
// Since PCDATA instructions have no width in the final code,
// it does not matter which phase we use for the update.
static int32
pctopcdata(Link *ctxt, LSym *sym, int32 oldval, Prog *p, int32 phase, void *arg)
{
	USED(sym);

	if(phase == 0 || p->as != ctxt->arch->APCDATA || p->from.offset != (uintptr)arg)
		return oldval;
	if((int32)p->to.offset != p->to.offset) {
		ctxt->diag("overflow in PCDATA instruction: %P", p);
		sysfatal("bad code");
	}
	return p->to.offset;
}

void
linkpcln(Link *ctxt, LSym *cursym)
{
	Prog *p;
	Pcln *pcln;
	int i, npcdata, nfuncdata, n;
	uint32 *havepc, *havefunc;

	ctxt->cursym = cursym;

	pcln = emallocz(sizeof *pcln);
	cursym->pcln = pcln;

	npcdata = 0;
	nfuncdata = 0;
	for(p = cursym->text; p != nil; p = p->link) {
		if(p->as == ctxt->arch->APCDATA && p->from.offset >= npcdata)
			npcdata = p->from.offset+1;
		if(p->as == ctxt->arch->AFUNCDATA && p->from.offset >= nfuncdata)
			nfuncdata = p->from.offset+1;
	}

	pcln->pcdata = emallocz(npcdata*sizeof pcln->pcdata[0]);
	pcln->npcdata = npcdata;
	pcln->funcdata = emallocz(nfuncdata*sizeof pcln->funcdata[0]);
	pcln->funcdataoff = emallocz(nfuncdata*sizeof pcln->funcdataoff[0]);
	pcln->nfuncdata = nfuncdata;

	funcpctab(ctxt, &pcln->pcsp, cursym, "pctospadj", pctospadj, nil);
	funcpctab(ctxt, &pcln->pcfile, cursym, "pctofile", pctofileline, pcln);
	funcpctab(ctxt, &pcln->pcline, cursym, "pctoline", pctofileline, nil);
	
	// tabulate which pc and func data we have.
	n = ((npcdata+31)/32 + (nfuncdata+31)/32)*4;
	havepc = emallocz(n);
	havefunc = havepc + (npcdata+31)/32;
	for(p = cursym->text; p != nil; p = p->link) {
		if(p->as == ctxt->arch->AFUNCDATA) {
			if((havefunc[p->from.offset/32]>>(p->from.offset%32))&1)
				ctxt->diag("multiple definitions for FUNCDATA $%d", p->from.offset);
			havefunc[p->from.offset/32] |= 1<<(p->from.offset%32);
		}
		if(p->as == ctxt->arch->APCDATA)
			havepc[p->from.offset/32] |= 1<<(p->from.offset%32);
	}
	// pcdata.
	for(i=0; i<npcdata; i++) {
		if(((havepc[i/32]>>(i%32))&1) == 0) 
			continue;
		funcpctab(ctxt, &pcln->pcdata[i], cursym, "pctopcdata", pctopcdata, (void*)(uintptr)i);
	}
	free(havepc);
	
	// funcdata
	if(nfuncdata > 0) {
		for(p = cursym->text; p != nil; p = p->link) {
			if(p->as == ctxt->arch->AFUNCDATA) {
				i = p->from.offset;
				pcln->funcdataoff[i] = p->to.offset;
				if(p->to.type != ctxt->arch->D_CONST) {
					// TODO: Dedup.
					//funcdata_bytes += p->to.sym->size;
					pcln->funcdata[i] = p->to.sym;
				}
			}
		}
	}
}

// iteration over encoded pcdata tables.

static uint32
getvarint(uchar **pp)
{
	uchar *p;
	int shift;
	uint32 v;

	v = 0;
	p = *pp;
	for(shift = 0;; shift += 7) {
		v |= (uint32)(*p & 0x7F) << shift;
		if(!(*p++ & 0x80))
			break;
	}
	*pp = p;
	return v;
}

void
pciternext(Pciter *it)
{
	uint32 v;
	int32 dv;

	it->pc = it->nextpc;
	if(it->done)
		return;
	if(it->p >= it->d.p + it->d.n) {
		it->done = 1;
		return;
	}

	// value delta
	v = getvarint(&it->p);
	if(v == 0 && !it->start) {
		it->done = 1;
		return;
	}
	it->start = 0;
	dv = (int32)(v>>1) ^ ((int32)(v<<31)>>31);
	it->value += dv;
	
	// pc delta
	v = getvarint(&it->p);
	it->nextpc = it->pc + v*it->pcscale;
}

void
pciterinit(Link *ctxt, Pciter *it, Pcdata *d)
{
	it->d = *d;
	it->p = it->d.p;
	it->pc = 0;
	it->nextpc = 0;
	it->value = -1;
	it->start = 1;
	it->done = 0;
	it->pcscale = ctxt->arch->minlc;
	pciternext(it);
}
