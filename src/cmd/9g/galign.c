// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include "gg.h"

int	thechar	= '9';
char*	thestring = "ppc64";
LinkArch*	thelinkarch;

void
linkarchinit(void)
{
	thestring = getgoarch();
	if(strcmp(thestring, "ppc64le") == 0)
		thelinkarch = &linkppc64le;
	else
		thelinkarch = &linkppc64;
}

vlong MAXWIDTH = 1LL<<50;

/*
 * go declares several platform-specific type aliases:
 * int, uint, float, and uintptr
 */
Typedef	typedefs[] =
{
	{"int",		TINT,		TINT64},
	{"uint",		TUINT,		TUINT64},
	{"uintptr",	TUINTPTR,	TUINT64},
	{0}
};

void
betypeinit(void)
{
	widthptr = 8;
	widthint = 8;
	widthreg = 8;

	zprog.link = P;
	zprog.as = AGOK;
	zprog.reg = NREG;
	zprog.from.name = D_NONE;
	zprog.from.type = D_NONE;
	zprog.from.reg = NREG;
	zprog.to = zprog.from;
	zprog.from3 = zprog.from;

	listinit9();
}
