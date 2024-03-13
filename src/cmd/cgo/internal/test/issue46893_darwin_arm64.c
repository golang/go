// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "_cgo_export.h"

void issue46893() {
    GoInt a = 0;
    GoString s;
    s.p = "test";
    s.n = 4;
    for (int i = 0; i < 10000000; i++) {
        goPanicIssue(&a, s);
    }
}
