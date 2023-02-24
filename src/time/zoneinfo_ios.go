// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ios

package time

import (
	"syscall"
)

var platformZoneSources []string // none on iOS

func gorootZoneSource(goroot string) (string, bool) {
	// The working directory at initialization is the root of the
	// app bundle: "/private/.../bundlename.app". That's where we
	// keep zoneinfo.zip for tethered iOS builds.
	// For self-hosted iOS builds, the zoneinfo.zip is in GOROOT.
	var roots []string
	if goroot != "" {
		roots = append(roots, goroot+"/lib/time")
	}
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
			return r + "/zoneinfo.zip", true
		}
	}
	return "", false
}

func initLocal() {
	// TODO(crawshaw): [NSTimeZone localTimeZone]
	localLoc = *UTC
}
