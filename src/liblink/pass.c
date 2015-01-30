// Inferno utils/6l/pass.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/pass.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
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

// Code and data passes.

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <link.h>

Prog*
brchain(Link *ctxt, Prog *p)
{
	int i;

	USED(ctxt);
	for(i=0; i<20; i++) {
		if(p == nil || p->as != AJMP || p->pcond == nil)
			return p;
		p = p->pcond;
	}
	return nil;
}

Prog*
brloop(Link *ctxt, Prog *p)
{
	int c;
	Prog *q;

	USED(ctxt);
	c = 0;
	for(q = p; q != nil; q = q->pcond) {
		if(q->as != AJMP || q->pcond == nil)
			break;
		c++;
		if(c >= 5000)
			return nil;
	}
	return q;
}

static void
checkaddr(Link *ctxt, Prog *p, Addr *a)
{
	// Check expected encoding, especially TYPE_CONST vs TYPE_ADDR.
	switch(a->type) {
	case TYPE_NONE:
		return;

	case TYPE_BRANCH:
		if(a->reg != 0 || a->index != 0 || a->scale != 0 || a->name != 0)
			break;
		return;

	case TYPE_TEXTSIZE:
		if(a->reg != 0 || a->index != 0 || a->scale != 0 || a->name != 0)
			break;
		return;

	case TYPE_MEM:
		//if(a->u.bits != 0)
		//	break;
		return;

	case TYPE_CONST:
		// TODO(rsc): After fixing SHRQ, check a->index != 0 too.
		if(a->name != 0 || a->sym != 0 || a->reg != 0) {
			ctxt->diag("argument %D is TYPE_CONST, should be TYPE_ADDR, in %P", a, p);
			return;
		}
		if(a->reg != 0 || a->scale != 0 || a->name != 0 || a->sym != nil || a->u.bits != 0)
			break;
		return;

	case TYPE_FCONST:
	case TYPE_SCONST:
		if(a->reg != 0 || a->index != 0 || a->scale != 0 || a->name != 0 || a->offset != 0 || a->sym != nil)
			break;
		return;

	case TYPE_REG:
		// TODO(rsc): After fixing PINSRQ, check a->offset != 0 too.
		// TODO(rsc): After fixing SHRQ, check a->index != 0 too.
		if(a->scale != 0 || a->name != 0 || a->sym != nil)
			break;
		return;

	case TYPE_ADDR:
		if(a->u.bits != 0)
			break;
		if(a->reg == 0 && a->index == 0 && a->scale == 0 && a->name == 0 && a->sym == nil)
			ctxt->diag("argument %D is TYPE_ADDR, should be TYPE_CONST, in %P", a, p);
		return;

	case TYPE_SHIFT:
		if(a->index != 0 || a->scale != 0 || a->name != 0 || a->sym != nil || a->u.bits != 0)
			break;
		return;

	case TYPE_REGREG:
		if(a->index != 0 || a->scale != 0 || a->name != 0 || a->sym != nil || a->u.bits != 0)
			break;
		return;

	case TYPE_REGREG2:
		return;

	case TYPE_INDIR:
		// Expect sym and name to be set, nothing else.
		// Technically more is allowed, but this is only used for *name(SB).
		if(a->reg != 0 || a->index != 0 || a->scale != 0 || a->name == 0 || a->offset != 0 || a->sym == nil || a->u.bits != 0)
			break;
		return;
	}

	ctxt->diag("invalid encoding for argument %D in %P", a, p);
}

void
linkpatch(Link *ctxt, LSym *sym)
{
	int32 c;
	char *name;
	Prog *p, *q;

	ctxt->cursym = sym;
	
	for(p = sym->text; p != nil; p = p->link) {
		checkaddr(ctxt, p, &p->from);
		checkaddr(ctxt, p, &p->from3);
		checkaddr(ctxt, p, &p->to);

		if(ctxt->arch->progedit)
			ctxt->arch->progedit(ctxt, p);
		if(p->to.type != TYPE_BRANCH)
			continue;
		if(p->to.u.branch != nil) {
			// TODO: Remove to.u.branch in favor of p->pcond.
			p->pcond = p->to.u.branch;
			continue;
		}
		if(p->to.sym != nil)
			continue;
		c = p->to.offset;
		for(q = sym->text; q != nil;) {
			if(c == q->pc)
				break;
			if(q->forwd != nil && c >= q->forwd->pc)
				q = q->forwd;
			else
				q = q->link;
		}
		if(q == nil) {
			name = "<nil>";
			if(p->to.sym)
				name = p->to.sym->name;
			ctxt->diag("branch out of range (%#ux)\n%P [%s]", c, p, name);
			p->to.type = TYPE_NONE;
		}
		p->to.u.branch = q;
		p->pcond = q;
	}
	
	for(p = sym->text; p != nil; p = p->link) {
		p->mark = 0;	/* initialization for follow */
		if(p->pcond != nil) {
			p->pcond = brloop(ctxt, p->pcond);
			if(p->pcond != nil)
			if(p->to.type == TYPE_BRANCH)
				p->to.offset = p->pcond->pc;
		}
	}
}
