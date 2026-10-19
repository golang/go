// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package help

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
)

// helpVerb is the executable help invocation for the current program:
// "logos help" in logos mode, otherwise "go help".
func helpVerb() string {
	if cfg.Logos {
		return "logos help"
	}
	return "go help"
}

// showHelpTopics reports whether a command listing includes the
// additional help topics section. The renamed logos root keeps it.
func showHelpTopics(cmd *base.Command) bool {
	if cmd.UsageLine == "go" {
		return true
	}
	return cfg.Logos && cmd.UsageLine == "logos"
}
