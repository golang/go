// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdio.h>
#include "issue4339.h"

static void
impl(void)
{
	//printf("impl\n");
}

Issue4339 exported4339 = {"bar", impl};

void
handle4339(Issue4339 *x)
{
	//printf("handle\n");
	x->bar();
	//printf("done\n");
}
