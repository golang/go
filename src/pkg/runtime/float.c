// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

// used as float64 via runtime· names
uint64	·nan		= 0x7FF8000000000001ULL;
uint64	·posinf	= 0x7FF0000000000000ULL;
uint64	·neginf	= 0xFFF0000000000000ULL;
