// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package help

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
)

func helpVerb() string {
	if cfg.Logos {
		return "logos help"
	}
	return "go help"
}

func showHelpTopics(cmd *base.Command) bool {
	if cmd.UsageLine == "go" {
		return true
	}
	return cfg.Logos && cmd.UsageLine == "logos"
}
