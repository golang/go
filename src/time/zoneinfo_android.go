// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parse the "tzdata" packed timezone file used on Android.
// The format is lifted from ZoneInfoDB.java and ZoneInfo.java in
// java/libcore/util in the AOSP.

package time

import (
	"runtime"
)

var zoneSources = []string{
	"/system/usr/share/zoneinfo/tzdata",
	"/data/misc/zoneinfo/current/tzdata",
	runtime.GOROOT() + "/lib/time/zoneinfo.zip",
}

func initLocal() {
	// TODO(elias.naur): getprop persist.sys.timezone
	localLoc = *UTC
}
