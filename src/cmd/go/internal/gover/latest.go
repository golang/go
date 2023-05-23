// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gover

import (
	"internal/goversion"
	"runtime"
	"strconv"
	"strings"
)

// TestVersion is initialized in the go command test binary
// to be $TESTGO_VERSION, to allow tests to override the
// go command's idea of its own version as returned by Local.
var TestVersion string

// Local returns the local Go version, the one implemented by this go command.
func Local() string {
	v := runtime.Version()
	if TestVersion != "" {
		v = TestVersion
	}
	if strings.HasPrefix(v, "go") {
		v := strings.TrimPrefix(v, "go")
		// Some builds use custom suffixes; strip them.
		if i := strings.IndexAny(v, " \t+-"); i >= 0 {
			v = v[:i]
		}
		if IsValid(v) {
			return v
		}
	}
	// Development branch. Use "Dev" version with just 1.N, no rc1 or .0 suffix.
	return "1." + strconv.Itoa(goversion.Version)
}
