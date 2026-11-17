// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"runtime"
	"strings"

	"cmd/go/internal/cfg"
	"cmd/go/internal/work"
)

func logosAppendTestEnv(env []string) []string {
	if !cfg.Logos {
		return env
	}
	// Tests can invoke this toolchain's go command to build a plugin. Its
	// experiment must match the test binary, including logos defaults.
	env = append(env, "GOEXPERIMENT="+cfg.RawGOEXPERIMENT)
	return append(env, work.LogosTargetArchEnv()...)
}

func logosInheritedTestEnv(name string) (string, bool) {
	if cfg.Logos && (name == "GOEXPERIMENT" || runtime.GOOS == "windows" && strings.EqualFold(name, "GOEXPERIMENT")) {
		return cfg.RawGOEXPERIMENT, true
	}
	for i := len(cfg.OrigEnv) - 1; i >= 0; i-- {
		key, value, ok := strings.Cut(cfg.OrigEnv[i], "=")
		if ok && (key == name || runtime.GOOS == "windows" && strings.EqualFold(key, name)) {
			return value, true
		}
	}
	return "", false
}
