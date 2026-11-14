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
	return append(env, work.LogosTargetArchEnv()...)
}

func logosInheritedTestEnv(name string) (string, bool) {
	for i := len(cfg.OrigEnv) - 1; i >= 0; i-- {
		key, value, ok := strings.Cut(cfg.OrigEnv[i], "=")
		if ok && (key == name || runtime.GOOS == "windows" && strings.EqualFold(key, name)) {
			return value, true
		}
	}
	return "", false
}
