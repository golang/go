// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "p.h"
#include "libgo.h"

extern int install_handler();
extern int check_handler();

int main(void) {
	int32_t res;

	int r1 = install_handler();
	if (r1!=0) {
		return r1;
	}

	if (!DidInitRun()) {
		fprintf(stderr, "ERROR: buildmode=c-archive init should run\n");
		return 2;
	}

	if (DidMainRun()) {
		fprintf(stderr, "ERROR: buildmode=c-archive should not run main\n");
		return 2;
	}

	int r2 = check_handler();
	if (r2!=0) {
		return r2;
	}

	res = FromPkg();
	if (res != 1024) {
		fprintf(stderr, "ERROR: FromPkg()=%d, want 1024\n", res);
		return 2;
	}

	CheckArgs();

	fprintf(stderr, "PASS\n");
	return 0;
}
