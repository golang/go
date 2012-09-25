// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include "gg.h"

int	thechar	= '6';
char*	thestring	= "amd64";

vlong MAXWIDTH = 1LL<<50;

/*
 * go declares several platform-specific type aliases:
 * int, uint, float, and uintptr
 */
Typedef	typedefs[] =
{
	"int",		TINT,		TINT64,
	"uint",		TUINT,		TUINT64,
	"uintptr",	TUINTPTR,	TUINT64,
	0
};

void
betypeinit(void)
{
	widthptr = 8;
	widthint = 8;

	zprog.link = P;
	zprog.as = AGOK;
	zprog.from.type = D_NONE;
	zprog.from.index = D_NONE;
	zprog.from.scale = 0;
	zprog.to = zprog.from;

	listinit();
}
