// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.18
// +build !go1.18

package debug

import (
	"runtime"
	"runtime/debug"
)

type BuildInfo struct {
	debug.BuildInfo
	GoVersion string // Version of Go that produced this binary
}

func readBuildInfo() (*BuildInfo, bool) {
	rinfo, ok := debug.ReadBuildInfo()
	if !ok {
		return nil, false
	}
	return &BuildInfo{
		GoVersion: runtime.Version(),
		BuildInfo: *rinfo,
	}, true
}
