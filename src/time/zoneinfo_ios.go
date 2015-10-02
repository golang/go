// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin
// +build arm arm64

package time

import "syscall"

var zoneFile string

func init() {
	wd, err := syscall.Getwd()
	if err != nil {
		return
	}

	// The working directory at initialization is the root of the
	// app bundle: "/private/.../bundlename.app". That's where we
	// keep zoneinfo.zip.
	zoneFile = wd + "/zoneinfo.zip"
}

func forceZipFileForTesting(zipOnly bool) {
	// On iOS we only have the zip file.
}

func initTestingZone() {
	z, err := loadZoneFile(zoneFile, "America/Los_Angeles")
	if err != nil {
		panic("cannot load America/Los_Angeles for testing: " + err.Error())
	}
	z.name = "Local"
	localLoc = *z
}

func initLocal() {
	// TODO(crawshaw): [NSTimeZone localTimeZone]
	localLoc = *UTC
}

func loadLocation(name string) (*Location, error) {
	z, err := loadZoneFile(zoneFile, name)
	if err != nil {
		return nil, err
	}
	z.name = name
	return z, nil
}
