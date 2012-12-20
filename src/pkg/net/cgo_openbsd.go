// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

/*
#include <netdb.h>
*/
import "C"

func cgoAddrInfoFlags() C.int {
	return C.AI_CANONNAME
}
