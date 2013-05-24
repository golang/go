// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "_cgo_export.h"

static void clobber_stack() {
	volatile char a[1024];
	int i;
	for(i = 0; i < sizeof a; i++)
		a[i] = 0xff;
}

static int call_go() {
	GoString s;
	s.p = "test";
	s.n = 4;
	return issue5548FromC(s, 42);
}

int issue5548_in_c() {
	clobber_stack();
	return call_go();
}
