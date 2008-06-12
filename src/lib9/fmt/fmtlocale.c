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
 * Fill in the internationalization stuff in the State structure.
 * For nil arguments, provide the sensible defaults:
 *	decimal is a period
 *	thousands separator is a comma
 *	thousands are marked every three digits
 */
void
fmtlocaleinit(Fmt *f, char *decimal, char *thousands, char *grouping)
{
	if(decimal == nil || decimal[0] == '\0')
		decimal = ".";
	if(thousands == nil)
		thousands = ",";
	if(grouping == nil)
		grouping = "\3";
	f->decimal = decimal;
	f->thousands = thousands;
	f->grouping = grouping;
}

/*
 * We are about to emit a digit in e.g. %'d.  If that digit would
 * overflow a thousands (e.g.) grouping, tell the caller to emit
 * the thousands separator.  Always advance the digit counter
 * and pointer into the grouping descriptor.
 */
int
__needsep(int *ndig, char **grouping)
{
	int group;

	(*ndig)++;
	group = *(unsigned char*)*grouping;
	/* CHAR_MAX means no further grouping. \0 means we got the empty string */
	if(group == 0xFF || group == 0x7f || group == 0x00)
		return 0;
	if(*ndig > group){
		/* if we're at end of string, continue with this grouping; else advance */
		if((*grouping)[1] != '\0')
			(*grouping)++;
		*ndig = 1;
		return 1;
	}
	return 0;
}

