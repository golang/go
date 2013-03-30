// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define Runemax Plan9Runemax
#include "/sys/include/libc.h"
#undef Runemax
#include "/sys/include/ctype.h"

enum
{
	Runemax = 0x10FFFF, /* maximum rune value */
};

char*	getgoos(void);
char*	getgoarch(void);
char*	getgoroot(void);
char*	getgoversion(void);
char*	getgoarm(void);
char*	getgo386(void);
char*	getgoextlinkenabled(void);

void	flagcount(char*, char*, int*);
void	flagint32(char*, char*, int32*);
void	flagint64(char*, char*, int64*);
void	flagstr(char*, char*, char**);
void	flagparse(int*, char***, void (*usage)(void));
void	flagfn0(char*, char*, void(*fn)(void));
void	flagfn1(char*, char*, void(*fn)(char*));
void	flagfn2(char*, char*, void(*fn)(char*, char*));
void	flagprint(int);
