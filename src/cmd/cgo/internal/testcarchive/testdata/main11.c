// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that the runtime reports faults whose saved registers do not
// authenticate. Each test crashes the program.

#include <stdio.h>
#include <string.h>

#include "libgo11.h"

int main(int argc, char **argv) {
	if (argc != 2) {
		fprintf(stderr, "usage: %s badpc|badfp|signedlr\n", argv[0]);
		return 2;
	}
	if (strcmp(argv[1], "badpc") == 0) {
		CallBadFuncValue();
	} else if (strcmp(argv[1], "badfp") == 0) {
		FaultWithBadFP();
	} else if (strcmp(argv[1], "signedlr") == 0) {
		FaultAfterPacibsp();
	} else {
		fprintf(stderr, "unknown test %s\n", argv[1]);
		return 2;
	}
	fprintf(stderr, "%s returned\n", argv[1]);
	return 2;
}
