// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package envcmd

import (
	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/settings"
)

func refreshLogosPolicy() {
	if !cfg.Logos || cfg.CmdName == "version" {
		return
	}
	s, _ := settings.Load(settings.CWDModuleRoot(), settings.Flags{Logos: true, Env: cfg.LogosStartupEnv()})
	cfg.ApplyLogosEnv(s)
}
