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

#include	"l.h"
#include	"../ld/lib.h"
#include "../../pkg/runtime/stack.h"

void
follow(void)
{
	LSym *s;

	if(debug['v'])
		Bprint(&bso, "%5.2f follow\n", cputime());
	Bflush(&bso);
	
	for(s = ctxt->textp; s != nil; s = s->next)
		ctxt->arch->follow(ctxt, s);
}

void
patch(void)
{
	LSym *s;

	if(debug['v'])
		Bprint(&bso, "%5.2f mkfwd\n", cputime());
	Bflush(&bso);
	for(s = ctxt->textp; s != nil; s = s->next)
		mkfwd(s);
	if(debug['v'])
		Bprint(&bso, "%5.2f patch\n", cputime());
	Bflush(&bso);

	if(flag_shared) {
		s = linklookup(ctxt, "init_array", 0);
		s->type = SINITARR;
		s->reachable = 1;
		s->hide = 1;
		addaddr(ctxt, s, linklookup(ctxt, INITENTRY, 0));
	}
	
	for(s = ctxt->textp; s != nil; s = s->next)
		linkpatch(ctxt, s);
}

void
dostkoff(void)
{
	LSym *s;

	for(s = ctxt->textp; s != nil; s = s->next)
		ctxt->arch->addstacksplit(ctxt, s);
}

void
span(void)
{
	LSym *s;

	if(debug['v'])
		Bprint(&bso, "%5.2f span\n", cputime());

	for(s = ctxt->textp; s != nil; s = s->next)
		ctxt->arch->assemble(ctxt, s);
}

void
pcln(void)
{
	LSym *s;

	for(s = ctxt->textp; s != nil; s = s->next)
		linkpcln(ctxt, s);
}
