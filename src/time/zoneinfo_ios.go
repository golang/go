// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ios

package time

import (
	"runtime"
	"syscall"
)

var zoneSources = []string{
	getZoneRoot() + "/zoneinfo.zip",
}

func getZoneRoot() string {
	// The working directory at initialization is the root of the
	// app bundle: "/private/.../bundlename.app". That's where we
	// keep zoneinfo.zip for tethered iOS builds.
	// For self-hosted iOS builds, the zoneinfo.zip is in GOROOT.
	roots := []string{runtime.GOROOT() + "/lib/time"}
	wd, err := syscall.Getwd()
	if err == nil {
		roots = append(roots, wd)
	}
	for _, r := range roots {
		var st syscall.Stat_t
		fd, err := syscall.Open(r, syscall.O_RDONLY, 0)
		if err != nil {
			continue
		}
		defer syscall.Close(fd)
		if err := syscall.Fstat(fd, &st); err == nil {
			return r
		}
	}
	return "/XXXNOEXIST"
}

func initLocal() {
	// TODO(crawshaw): [NSTimeZone localTimeZone]
	localLoc = *UTC
}
