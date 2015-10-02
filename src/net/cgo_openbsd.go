// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo,!netgo

package net

/*
#include <netdb.h>
*/
import "C"

const cgoAddrInfoFlags = C.AI_CANONNAME
