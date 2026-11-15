// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfg

import (
	"os"
	"strings"

	"cmd/go/internal/logos/settings"
)

func recomputeExperiment() {
	kept := BuildContext.ToolTags[:0:0]
	for _, tag := range BuildContext.ToolTags {
		if !strings.HasPrefix(tag, "goexperiment.") {
			kept = append(kept, tag)
		}
	}
	BuildContext.ToolTags = kept
	computeExperiment()
}

func logosApplyRawExperiment(raw string) {
	if raw == RawGOEXPERIMENT {
		return
	}
	RawGOEXPERIMENT = raw
	recomputeExperiment()
	os.Setenv("GOEXPERIMENT", RawGOEXPERIMENT)
	for i := range CmdEnv {
		if CmdEnv[i].Name == "GOEXPERIMENT" {
			CmdEnv[i].Value = RawGOEXPERIMENT
		}
	}
}

func RefreshLogosPreemptible() {
	if ExperimentErr != nil {
		return
	}
	if raw, ok := settings.PreemptibleDefault(RawGOEXPERIMENT, logosInitGOEXPERIMENT, Experiment.PreemptibleLoops, Logos, Goarch); ok {
		logosApplyRawExperiment(raw)
	}
}

func DropLogosExperiments() {
	os.Setenv("GOEXPERIMENT", settings.DropLogosExperiments(os.Getenv("GOEXPERIMENT")))
}
