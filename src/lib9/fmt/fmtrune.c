/*
 * The authors of this software are Rob Pike and Ken Thompson,
 * with contributions from Mike Burrows and Sean Dorward.
 *
 *     Copyright (c) 2002-2006 by Lucent Technologies.
 *     Portions Copyright (c) 2004 Google Inc.
 * 
 * Permission to use, copy, modify, and distribute this software for any
 * purpose without fee is hereby granted, provided that this entire notice
 * is included in all copies of any software which is or includes a copy
 * or modification of this software and in all copies of the supporting
 * documentation for such software.
 * THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTY.  IN PARTICULAR, NEITHER THE AUTHORS NOR LUCENT TECHNOLOGIES 
 * NOR GOOGLE INC MAKE ANY REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING 
 * THE MERCHANTABILITY OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
 */

#include <u.h>
#include <libc.h>
#include "fmtdef.h"

int
fmtrune(Fmt *f, int r)
{
	Rune *rt;
	char *t;
	int n;

	if(f->runes){
		rt = (Rune*)f->to;
		FMTRCHAR(f, rt, f->stop, r);
		f->to = rt;
		n = 1;
	}else{
		t = (char*)f->to;
		FMTRUNE(f, t, f->stop, r);
		n = t - (char*)f->to;
		f->to = t;
	}
	f->nfmt += n;
	return 0;
}
