// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !cmd_go_bootstrap

package telemetrystats

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/modload"
	"cmd/internal/telemetry"
)

func Increment() {
	incrementConfig()
	incrementVersionCounters()
}

// incrementConfig increments counters for the configuration
// the command is running in.
func incrementConfig() {
	if !modload.WillBeEnabled() {
		telemetry.Inc("go/mode:gopath")
	} else if workfile := modload.FindGoWork(base.Cwd()); workfile != "" {
		telemetry.Inc("go/mode:workspace")
	} else {
		telemetry.Inc("go/mode:module")
	}
	telemetry.Inc("go/platform/target/goos:" + cfg.Goos)
	telemetry.Inc("go/platform/target/goarch:" + cfg.Goarch)
	switch cfg.Goarch {
	case "386":
		telemetry.Inc("go/platform/target/go386:" + cfg.GO386)
	case "amd64":
		telemetry.Inc("go/platform/target/goamd64:" + cfg.GOAMD64)
	case "arm":
		telemetry.Inc("go/platform/target/goarm:" + cfg.GOARM)
	case "arm64":
		telemetry.Inc("go/platform/target/goarm64:" + cfg.GOARM64)
	case "mips":
		telemetry.Inc("go/platform/target/gomips:" + cfg.GOMIPS)
	case "ppc64":
		telemetry.Inc("go/platform/target/goppc64:" + cfg.GOPPC64)
	case "riscv64":
		telemetry.Inc("go/platform/target/goriscv64:" + cfg.GORISCV64)
	case "wasm":
		telemetry.Inc("go/platform/target/gowasm:" + cfg.GOWASM)
	}
}
