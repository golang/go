// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin
// +build arm arm64

package time

import "syscall"

var zoneSources = []string{
	getZipParent() + "/zoneinfo.zip",
}

func getZipParent() string {
	wd, err := syscall.Getwd()
	if err != nil {
		return "/XXXNOEXIST"
	}

	// The working directory at initialization is the root of the
	// app bundle: "/private/.../bundlename.app". That's where we
	// keep zoneinfo.zip.
	return wd
}

func initLocal() {
	// TODO(crawshaw): [NSTimeZone localTimeZone]
	localLoc = *UTC
}
