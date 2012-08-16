// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
// mingw32 on windows/386 provides usleep() but not sleep(),
// as we don't want to require all other OSes to provide usleep,
// we emulate sleep(int s) using win32 API Sleep(int ms).

#include <windows.h>

unsigned int sleep(unsigned int seconds) {
	Sleep(1000 * seconds);
	return 0;
}

*/
import "C"
