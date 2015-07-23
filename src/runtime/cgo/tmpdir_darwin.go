// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build arm arm64

package cgo

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework CoreFoundation -framework Foundation

#include <sys/param.h>
#include <CoreFoundation/CFString.h>
#include <Foundation/NSPathUtilities.h>

char tmpdir[MAXPATHLEN];

char* loadtmpdir() {
	tmpdir[0] = 0;
	CFStringRef path = (CFStringRef)NSTemporaryDirectory();
	CFStringGetCString(path, tmpdir, sizeof(tmpdir), kCFStringEncodingUTF8);
	return tmpdir;
}
*/
import "C"

func init() {
	if Getenv("TMPDIR") != "" {
		return
	}
	dir := C.GoString(C.loadtmpdir())
	if len(dir) == 0 {
		return
	}
	if dir[len(dir)-1] == '/' {
		dir = dir[:len(dir)-1]
	}
	Setenv("TMPDIR", dir)
}
