// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct { char *p; intmax_t n; } GoString;
extern signed char DidInitRun();
extern signed char DidMainRun();
extern GoString FromPkg();

int main(void) {
	GoString res;

	if (DidMainRun()) {
		fprintf(stderr, "ERROR: buildmode=c-archive should not run main\n");
		return 2;
	}

	if (!DidInitRun()) {
		fprintf(stderr, "ERROR: buildmode=c-archive init should run\n");
		return 2;
	}

	res = FromPkg();
	if (strcmp(res.p, "str")) {
		fprintf(stderr, "ERROR: FromPkg()='%s', want 'str'\n", res.p);
		return 2;
	}

	return 0;
}
