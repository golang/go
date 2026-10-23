// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test C function pointers that Go passes back to C.

#include <stdio.h>

#include "libgo13.h"

int main(void) {
	int x = Apply(21);
	if (x != 42) {
		fprintf(stderr, "Apply(21) = %d, want 42\n", x);
		return 2;
	}
	if (ContextCalls() == 0) {
		fprintf(stderr, "context function not called\n");
		return 2;
	}
	return 0;
}
