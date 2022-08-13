// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "_cgo_export.h"

int get8148(void) {
	T t;
	t.i = 42;
	return issue8148Callback(&t);
}
