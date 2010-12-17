// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <assert.h>
#include "life.h"
#include "_cgo_export.h"

const int MYCONST = 0;

// Do the actual manipulation of the life board in C.  This could be
// done easily in Go, we are just using C for demonstration
// purposes.
void
Step(int x, int y, int *a, int *n)
{
	struct GoStart_return r;

	// Use Go to start 4 goroutines each of which handles 1/4 of the
	// board.
	r = GoStart(0, x, y, 0, x / 2, 0, y / 2, a, n);
	assert(r.r0 == 0 && r.r1 == 100);	// test multiple returns
	r = GoStart(1, x, y, x / 2, x, 0, y / 2, a, n);
	assert(r.r0 == 1 && r.r1 == 101);	// test multiple returns
	GoStart(2, x, y, 0, x / 2, y / 2, y, a, n);
	GoStart(3, x, y, x / 2, x, y / 2, y, a, n);
	GoWait(0);
	GoWait(1);
	GoWait(2);
	GoWait(3);
}

// The actual computation.  This is called in parallel.
void
DoStep(int xdim, int ydim, int xstart, int xend, int ystart, int yend, int *a, int *n)
{
	int x, y, c, i, j;

	for(x = xstart; x < xend; x++) {
		for(y = ystart; y < yend; y++) {
			c = 0;
			for(i = -1; i <= 1; i++) {
				for(j = -1; j <= 1; j++) {
				  if(x+i >= 0 && x+i < xdim &&
					y+j >= 0 && y+j < ydim &&
					(i != 0 || j != 0))
				    c += a[(x+i)*xdim + (y+j)] != 0;
				}
			}
			if(c == 3 || (c == 2 && a[x*xdim + y] != 0))
				n[x*xdim + y] = 1;
			else
				n[x*xdim + y] = 0;
		}
	}
}
