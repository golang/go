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

	for(i=0; i<20; i++) {
		if(p == nil || p->as != ctxt->arch->AJMP || p->pcond == nil)
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

	c = 0;
	for(q = p; q != nil; q = q->pcond) {
		if(q->as != ctxt->arch->AJMP || q->pcond == nil)
			break;
		c++;
		if(c >= 5000)
			return nil;
	}
	return q;
}

void
linkpatch(Link *ctxt, LSym *sym)
{
	int32 c;
	Prog *p, *q;

	ctxt->cursym = sym;
	
	for(p = sym->text; p != nil; p = p->link) {
		if(ctxt->arch->progedit)
			ctxt->arch->progedit(ctxt, p);
		if(p->to.type != ctxt->arch->D_BRANCH)
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
			ctxt->diag("branch out of range (%#ux)\n%P [%s]",
				c, p, p->to.sym ? p->to.sym->name : "<nil>");
			p->to.type = ctxt->arch->D_NONE;
		}
		p->to.u.branch = q;
		p->pcond = q;
	}
	
	for(p = sym->text; p != nil; p = p->link) {
		p->mark = 0;	/* initialization for follow */
		if(p->pcond != nil) {
			p->pcond = brloop(ctxt, p->pcond);
			if(p->pcond != nil)
			if(p->to.type == ctxt->arch->D_BRANCH)
				p->to.offset = p->pcond->pc;
		}
	}
}
