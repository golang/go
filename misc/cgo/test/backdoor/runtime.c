// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Expose some runtime functions for testing.
// Must be in a non-cgo-using package so that
// the go command compiles this file with 6c, not gcc.

// +build gc

typedef char bool;

// This is what a cgo-compiled stub declaration looks like.
void
·Issue7695(struct{void *y[8*sizeof(void*)];}p)
{
	USED(p);
}
