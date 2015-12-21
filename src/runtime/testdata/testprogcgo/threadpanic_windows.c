// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdlib.h>
#include <stdio.h>

void gopanic(void);

static void*
die(void* x)
{
	gopanic();
	return 0;
}

void
start(void)
{
	if(_beginthreadex(0, 0, die, 0, 0, 0) != 0)
		printf("_beginthreadex failed\n");
}
