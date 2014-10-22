// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is compiled by cmd/dist to obtain debug information
// about the given header files.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"
#include "type.h"
#include "race.h"
#include "chan.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
