// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"flag"

	"cmd/compile/internal/base"
	"cmd/compile/internal/types2"
)

func logosConfigureComptime(conf *types2.Config, info *types2.Info) {
	if !base.Logos || base.LogosCfg == "" {
		return
	}
	policy := logosLintPolicy()
	conf.LogosCTFEMaxSteps = policy.Thresholds.MaxCTFESteps
	info.LogosCTFEMaxSteps = conf.LogosCTFEMaxSteps
	if !logosComptimeNodesExplicit() {
		base.LogosComptimeNodes = policy.Thresholds.MaxComptimeASTNodes
	}
	conf.LogosComptimeNodeLimit = base.LogosComptimeNodes
}

func logosComptimeNodesExplicit() bool {
	explicit := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "logos.comptime_nodes" {
			explicit = true
		}
	})
	return explicit
}
