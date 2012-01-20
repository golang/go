// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
#include <windows.h>

unsigned int sleep(unsigned int seconds) {
	Sleep(1000 * seconds);
	return 0;
}

*/
import "C"
