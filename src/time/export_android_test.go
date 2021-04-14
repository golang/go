// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

func ForceAndroidTzdataForTest(tzdata bool) {
	forceZipFileForTesting(false)
	if tzdata {
		zoneSources = zoneSources[:len(zoneSources)-1]
	}
}
