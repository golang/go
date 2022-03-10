// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !ios && !android

package time

func gorootZoneSource(goroot string) (string, bool) {
	if goroot == "" {
		return "", false
	}
	return goroot + "/lib/time/zoneinfo.zip", true
}
