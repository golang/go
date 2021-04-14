// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "_cgo_export.h"

#if defined(WIN32) || defined(_AIX)
extern void setCallback(void *);
void init() {
	setCallback(goCallback);
}
#else
void init() {}
#endif
