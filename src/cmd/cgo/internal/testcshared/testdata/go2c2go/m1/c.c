// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "libtestgo2c2go.h"

int CFunc(void) {
	return (GoFunc() << 8) + 2;
}
