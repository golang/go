/*
 * The authors of this software are Rob Pike and Ken Thompson.
 *              Copyright (c) 1998-2002 by Lucent Technologies.
 * Permission to use, copy, modify, and distribute this software for any
 * purpose without fee is hereby granted, provided that this entire notice
 * is included in all copies of any software which is or includes a copy
 * or modification of this software and in all copies of the supporting
 * documentation for such software.
 * THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTY.  IN PARTICULAR, NEITHER THE AUTHORS NOR LUCENT TECHNOLOGIES MAKE ANY
 * REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY
 * OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
 */

/*
 * compiler directive on Plan 9
 */
#ifndef USED
#define USED(x) if(x);else
#endif

/*
 * easiest way to make sure these are defined
 */
#define uchar	_fmtuchar
#define ushort	_fmtushort
#define uint	_fmtuint
#define ulong	_fmtulong
#define vlong	_fmtvlong
#define uvlong	_fmtuvlong
typedef unsigned char		uchar;
typedef unsigned short		ushort;
typedef unsigned int		uint;
typedef unsigned long		ulong;
typedef unsigned long long	uvlong;
typedef long long		vlong;

/*
 * nil cannot be ((void*)0) on ANSI C,
 * because it is used for function pointers
 */
#undef	nil
#define	nil	0

#undef	nelem
#define	nelem	((void*)0)

