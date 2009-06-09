// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "l.h"
#include "compat.h"
enum
{
	PtrSize = 4
};
#define pcond cond
#include "../ld/go.c"
