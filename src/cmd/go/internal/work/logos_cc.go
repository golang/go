// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"cmp"

	"cmd/go/internal/cfg"
)

func logosCompilerFallback(key, def string) string {
	if key == "CXX" {
		return cfg.ZigCXX(def)
	}
	return cfg.ZigCC(def)
}

func logosCgoToolEnv() []string {
	var env []string
	cc := cmp.Or(cfg.Getenv("CC"), cfg.DefaultCC(cfg.Goos, cfg.Goarch))
	if fb := cfg.ZigCC(cc); fb != cc {
		env = append(env, "CC="+fb)
	}
	cxx := cmp.Or(cfg.Getenv("CXX"), cfg.DefaultCXX(cfg.Goos, cfg.Goarch))
	if fb := cfg.ZigCXX(cxx); fb != cxx {
		env = append(env, "CXX="+fb)
	}
	return env
}
