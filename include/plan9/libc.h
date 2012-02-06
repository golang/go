// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "/sys/include/libc.h"
#include "/sys/include/ctype.h"

enum
{
	Runemax = 0x10FFFF, /* maximum rune value */
};

char*	getgoos(void);
char*	getgoarch(void);
char*	getgoroot(void);
char*	getgoversion(void);
