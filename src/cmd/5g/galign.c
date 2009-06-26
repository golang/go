// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "gg.h"

int	thechar	= '5';
char*	thestring	= "arm";


/*
 * go declares several platform-specific type aliases:
 * int, uint, float, and uintptr
 */
Typedef	typedefs[] =
{
	"int",		TINT,		TINT32,
	"uint",		TUINT,		TUINT32,
	"uintptr",	TUINTPTR,	TUINT32,
	"float",	TFLOAT,		TFLOAT32,
	0
};

void
betypeinit(void)
{
	maxround = 4;
	widthptr = 4;

	zprog.link = P;
	zprog.as = AGOK;
	zprog.scond = C_SCOND_NONE;
	zprog.reg = NREG;
	zprog.from.type = D_NONE;
	zprog.from.index = D_NONE;
	zprog.from.reg = NREG;
	zprog.from.scale = 0;
	zprog.to = zprog.from;

	listinit();
}
