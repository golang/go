// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !cmd_go_bootstrap

package telemetrystats

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/modload"
	"cmd/internal/telemetry/counter"
	"strings"
)

func Increment() {
	incrementConfig()
	incrementVersionCounters()
}

// incrementConfig increments counters for the configuration
// the command is running in.
func incrementConfig() {
	if !modload.WillBeEnabled(modload.LoaderState) {
		counter.Inc("go/mode:gopath")
	} else if workfile := modload.FindGoWork(modload.LoaderState, base.Cwd()); workfile != "" {
		counter.Inc("go/mode:workspace")
	} else {
		counter.Inc("go/mode:module")
	}
	counter.Inc("go/platform/target/goos:" + cfg.Goos)
	counter.Inc("go/platform/target/goarch:" + cfg.Goarch)
	switch cfg.Goarch {
	case "386":
		counter.Inc("go/platform/target/go386:" + cfg.GO386)
	case "amd64":
		counter.Inc("go/platform/target/goamd64:" + cfg.GOAMD64)
	case "arm":
		counter.Inc("go/platform/target/goarm:" + cfg.GOARM)
	case "arm64":
		counter.Inc("go/platform/target/goarm64:" + cfg.GOARM64)
	case "mips":
		counter.Inc("go/platform/target/gomips:" + cfg.GOMIPS)
	case "ppc64":
		counter.Inc("go/platform/target/goppc64:" + cfg.GOPPC64)
	case "riscv64":
		counter.Inc("go/platform/target/goriscv64:" + cfg.GORISCV64)
	case "wasm":
		counter.Inc("go/platform/target/gowasm:" + cfg.GOWASM)
	}

	// Use cfg.Experiment.String instead of cfg.Experiment.Enabled
	// because we only want to count the experiments that differ
	// from the baseline.
	if cfg.Experiment != nil {
		for exp := range strings.SplitSeq(cfg.Experiment.String(), ",") {
			if exp == "" {
				continue
			}
			counter.Inc("go/goexperiment:" + exp)
		}
	}
}
