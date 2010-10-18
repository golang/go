// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

extern int32 write(int32 fd, void* buffer, int32 nbytes);
extern void exits(int8* msg);
extern int32 brk_(void*);

/* rfork */
enum
{
	RFNAMEG         = (1<<0),
	RFENVG          = (1<<1),
	RFFDG           = (1<<2),
	RFNOTEG         = (1<<3),
	RFPROC          = (1<<4),
	RFMEM           = (1<<5),
	RFNOWAIT        = (1<<6),
	RFCNAMEG        = (1<<10),
	RFCENVG         = (1<<11),
	RFCFDG          = (1<<12),
	RFREND          = (1<<13),
	RFNOMNT         = (1<<14)
};
extern int32 rfork(int32 flags, void *stk, M *m, G *g, void (*fn)(void));
extern int32 plan9_semacquire(uint32 *addr, int32 block);
extern int32 plan9_semrelease(uint32 *addr, int32 count);
