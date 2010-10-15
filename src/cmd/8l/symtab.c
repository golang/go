// Inferno utils/8l/span.c
// http://code.google.com/p/inferno-os/source/browse/utils/8l/span.c
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

// Symbol table.

#include	"l.h"
#include	"../ld/lib.h"

void
putsymb(char *s, int t, int32 v, int ver, Sym *go)
{
	int i, f;
	vlong gv;

	if(t == 'f')
		s++;
	lput(v);
	if(ver)
		t += 'a' - 'A';
	cput(t+0x80);			/* 0x80 is variable length */

	if(t == 'Z' || t == 'z') {
		cput(s[0]);
		for(i=1; s[i] != 0 || s[i+1] != 0; i += 2) {
			cput(s[i]);
			cput(s[i+1]);
		}
		cput(0);
		cput(0);
		i++;
	}
	else {
		for(i=0; s[i]; i++)
			cput(s[i]);
		cput(0);
	}
	gv = 0;
	if(go) {
		if(!go->reachable)
			sysfatal("unreachable type %s", go->name);
		gv = go->value+INITDAT;
	}
	lput(gv);

	symsize += 4 + 1 + i+1 + 4;

	if(debug['n']) {
		if(t == 'z' || t == 'Z') {
			Bprint(&bso, "%c %.8ux ", t, v);
			for(i=1; s[i] != 0 || s[i+1] != 0; i+=2) {
				f = ((s[i]&0xff) << 8) | (s[i+1]&0xff);
				Bprint(&bso, "/%x", f);
			}
			Bprint(&bso, "\n");
			return;
		}
		if(ver)
			Bprint(&bso, "%c %.8ux %s<%d> %s (%.8llux)\n", t, v, s, ver, go ? go->name : "", gv);
		else
			Bprint(&bso, "%c %.8ux %s\n", t, v, s, go ? go->name : "", gv);
	}
}

void
asmsym(void)
{
	Auto *a;
	Sym *s;
	int h;

	s = lookup("etext", 0);
	if(s->type == STEXT)
		putsymb(s->name, 'T', s->value, s->version, 0);

	for(h=0; h<NHASH; h++)
		for(s=hash[h]; s!=S; s=s->hash)
			switch(s->type) {
			case SCONST:
			case SRODATA:
			case SDATA:
			case SELFDATA:
				if(!s->reachable)
					continue;
				putsymb(s->name, 'D', symaddr(s), s->version, s->gotype);
				continue;

			case SMACHO:
				if(!s->reachable)
					continue;
				putsymb(s->name, 'D', s->value+INITDAT+segdata.filelen-dynptrsize, s->version, s->gotype);
				continue;

			case SBSS:
				if(!s->reachable)
					continue;
				putsymb(s->name, 'B', s->value+INITDAT, s->version, s->gotype);
				continue;

			case SFIXED:
				putsymb(s->name, 'B', s->value, s->version, s->gotype);
				continue;

			case SFILE:
				putsymb(s->name, 'f', s->value, s->version, 0);
				continue;
			}

	for(s = textp; s != nil; s = s->next) {
		/* filenames first */
		for(a=s->autom; a; a=a->link)
			if(a->type == D_FILE)
				putsymb(a->asym->name, 'z', a->aoffset, 0, 0);
			else
			if(a->type == D_FILE1)
				putsymb(a->asym->name, 'Z', a->aoffset, 0, 0);

		putsymb(s->name, 'T', s->value, s->version, s->gotype);

		/* frame, auto and param after */
		putsymb(".frame", 'm', s->text->to.offset+4, 0, 0);

		for(a=s->autom; a; a=a->link)
			if(a->type == D_AUTO)
				putsymb(a->asym->name, 'a', -a->aoffset, 0, a->gotype);
			else
			if(a->type == D_PARAM)
				putsymb(a->asym->name, 'p', a->aoffset, 0, a->gotype);
	}
	if(debug['v'] || debug['n'])
		Bprint(&bso, "symsize = %ud\n", symsize);
	Bflush(&bso);
}
