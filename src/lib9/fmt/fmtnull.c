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

/*
 * Absorb output without using resources.
 */
static Rune nullbuf[32];

static int
__fmtnullflush(Fmt *f)
{
	f->to = nullbuf;
	f->nfmt = 0;
	return 0;
}

int
fmtnullinit(Fmt *f)
{
	memset(f, 0, sizeof *f);
	f->runes = 1;
	f->start = nullbuf;
	f->to = nullbuf;
	f->stop = nullbuf+nelem(nullbuf);
	f->flush = __fmtnullflush;
	fmtlocaleinit(f, nil, nil, nil);
	return 0;
}

