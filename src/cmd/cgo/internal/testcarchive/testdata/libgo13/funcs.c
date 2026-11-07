// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

int twice(int x) { return 2 * x; }

int apply(int (*f)(int), int x) { return f(x); }

static int calls;

void contextFunc(void *arg) { calls++; }

int contextCalls(void) { return calls; }
