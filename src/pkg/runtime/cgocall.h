// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Cgo interface.
 */

void runtime·cgocall(void (*fn)(void*), void*);
void runtime·cgocallback(void (*fn)(void), void*, uintptr);
void *runtime·cmalloc(uintptr);
void runtime·cfree(void*);
