// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include "gg.h"

int	thechar	= '6';
char*	thestring	= "amd64";
LinkArch*	thelinkarch = &linkamd64;

void
linkarchinit(void)
{
	if(strcmp(getgoarch(), "amd64p32") == 0)
		thelinkarch = &linkamd64p32;
}

vlong MAXWIDTH = 1LL<<50;

int	addptr = AADDQ;
int	movptr = AMOVQ;
int	leaptr = ALEAQ;
int	cmpptr = ACMPQ;

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
	if(strcmp(getgoarch(), "amd64p32") == 0) {
		widthptr = 4;
		widthint = 4;
		addptr = AADDL;
		movptr = AMOVL;
		leaptr = ALEAL;
		cmpptr = ACMPL;
		typedefs[0].sameas = TINT32;
		typedefs[1].sameas = TUINT32;
		typedefs[2].sameas = TUINT32;
		
	}

	zprog.link = P;
	zprog.as = AGOK;
	zprog.from.type = D_NONE;
	zprog.from.index = D_NONE;
	zprog.from.scale = 0;
	zprog.to = zprog.from;

	listinit6();
}
