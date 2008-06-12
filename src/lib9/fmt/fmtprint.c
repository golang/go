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
 * format a string into the output buffer
 * designed for formats which themselves call fmt,
 * but ignore any width flags
 */
int
fmtprint(Fmt *f, char *fmt, ...)
{
	va_list va;
	int n;

	f->flags = 0;
	f->width = 0;
	f->prec = 0;
	VA_COPY(va, f->args);
	VA_END(f->args);
	va_start(f->args, fmt);
	n = dofmt(f, fmt);
	va_end(f->args);
	f->flags = 0;
	f->width = 0;
	f->prec = 0;
	VA_COPY(f->args,va);
	VA_END(va);
	if(n >= 0)
		return 0;
	return n;
}

