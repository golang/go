// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"cmd/go/internal/cfg"
)

const logosLoopStrideFlag = "-d=logosloopstride=4096"

func logosApplyLoopStride() {
	if !cfg.Logos || cfg.Goarch != "wasm" {
		return
	}
	if cfg.Experiment == nil || !cfg.Experiment.PreemptibleLoops {
		return
	}
	forcedGcflags = append(forcedGcflags, logosLoopStrideFlag)
}
