// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Cgo interface.
 * Dynamically linked shared libraries compiled with gcc
 * know these data structures and functions too.
 * See ../../libcgo/cgocall.c
 */

typedef struct CgoWork CgoWork;
typedef struct CgoServer CgoServer;
typedef struct Cgo Cgo;

struct Cgo
{
	Lock lock;
	CgoServer *idle;
	CgoWork *whead;
	CgoWork *wtail;
};

struct CgoServer
{
	CgoServer *next;
	Note note;
	CgoWork *work;
};

struct CgoWork
{
	CgoWork *next;
	Note note;
	void (*fn)(void*);
	void *arg;
};

void cgocall(void (*fn)(void*), void*);

void *cmalloc(uintptr);
void cfree(void*);
