// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is not in bigstack_windows.c because it needs to be part of
// testprogcgo but is not part of the DLL built from bigstack_windows.c.

#include "_cgo_export.h"

void CallGoBigStack1(char* p) {
	goBigStack1(p);
}
