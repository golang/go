// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Cgo interface.
 */

void cgocall(void (*fn)(void*), void*);
void cgocallback(void (*fn)(void), void*, int32);
void *cmalloc(uintptr);
void cfree(void*);
