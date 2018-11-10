// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 5227: linker incorrectly treats common symbols and
// leaves them undefined.

package cgotest

/*
typedef struct {
        int Count;
} Fontinfo;

Fontinfo SansTypeface;

extern void init();

Fontinfo loadfont() {
        Fontinfo f = {0};
        return f;
}

void init() {
        SansTypeface = loadfont();
}
*/
import "C"

import "testing"

func test5227(t *testing.T) {
	C.init()
}

func selectfont() C.Fontinfo {
	return C.SansTypeface
}
