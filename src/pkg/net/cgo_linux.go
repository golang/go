// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

/*
#include <netdb.h>
*/
import "C"

func cgoAddrInfoMask() C.int {
	return C.AI_CANONNAME | C.AI_V4MAPPED | C.AI_ALL
}
