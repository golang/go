// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdint.h>
#include <stdio.h>

extern signed char DidInitRun();
extern signed char DidMainRun();
extern int32_t FromPkg();

int main(void) {
	int32_t res;

	if (!DidInitRun()) {
		fprintf(stderr, "ERROR: buildmode=c-archive init should run\n");
		return 2;
	}

	if (DidMainRun()) {
		fprintf(stderr, "ERROR: buildmode=c-archive should not run main\n");
		return 2;
	}

	res = FromPkg();
	if (res != 1024) {
		fprintf(stderr, "ERROR: FromPkg()=%d, want 1024\n", res);
		return 2;
	}

	return 0;
}
