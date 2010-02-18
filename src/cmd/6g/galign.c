// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "gg.h"

int	thechar	= '6';
char*	thestring	= "amd64";


/*
 * go declares several platform-specific type aliases:
 * int, uint, float, and uintptr
 */
Typedef	typedefs[] =
{
	"int",		TINT,		TINT32,
	"uint",		TUINT,		TUINT32,
	"uintptr",	TUINTPTR,	TUINT64,
	"float",	TFLOAT,		TFLOAT32,
	"complex",	TCOMPLEX,	TCOMPLEX64,
	0
};

void
betypeinit(void)
{
	maxround = 8;
	widthptr = 8;

	zprog.link = P;
	zprog.as = AGOK;
	zprog.from.type = D_NONE;
	zprog.from.index = D_NONE;
	zprog.from.scale = 0;
	zprog.to = zprog.from;

	listinit();
}
