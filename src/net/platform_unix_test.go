// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || js || wasip1

package net

import (
	"os/exec"
	"runtime"
	"strconv"
)

var unixEnabledOnAIX bool

func init() {
	if runtime.GOOS == "aix" {
		// Unix network isn't properly working on AIX 7.2 with
		// Technical Level < 2.
		// The information is retrieved only once in this init()
		// instead of everytime testableNetwork is called.
		out, _ := exec.Command("oslevel", "-s").Output()
		if len(out) >= len("7200-XX-ZZ-YYMM") { // AIX 7.2, Tech Level XX, Service Pack ZZ, date YYMM
			aixVer := string(out[:4])
			tl, _ := strconv.Atoi(string(out[5:7]))
			unixEnabledOnAIX = aixVer > "7200" || (aixVer == "7200" && tl >= 2)
		}
	}
}

func supportsUnixSocket() bool {
	switch runtime.GOOS {
	case "android", "ios":
		return false
	case "aix":
		return unixEnabledOnAIX
	}
	return true
}
