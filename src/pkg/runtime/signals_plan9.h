// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define N SigNotify
#define T SigThrow
#define P SigPanic

SigTab runtime·sigtab[] = {
	P, "sys: fp:",

	// Go libraries expect to be able
	// to recover from memory
	// read/write errors, so we flag
	// those as panics. All other traps
	// are generally more serious and
	// should immediately throw an
	// exception.
	P, "sys: trap: fault read addr",
	P, "sys: trap: fault write addr",
	T, "sys: trap:",

	N, "sys: bad sys call",
};

#undef N
#undef T
#undef P
