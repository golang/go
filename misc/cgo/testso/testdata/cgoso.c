// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "_cgo_export.h"

#ifdef WIN32
extern void setCallback(void *);
void init() {
	setCallback(goCallback);
}
#else
void init() {}
#endif
