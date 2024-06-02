// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !cmd_go_bootstrap && windows

package telemetrystats

import (
	"fmt"

	"cmd/internal/telemetry"

	"golang.org/x/sys/windows"
)

func incrementVersionCounters() {
	v := windows.RtlGetVersion()
	telemetry.Inc(fmt.Sprintf("go/platform/host/windows/major-version:%d", v.MajorVersion))
	telemetry.Inc(fmt.Sprintf("go/platform/host/windows/version:%d-%d", v.MajorVersion, v.MinorVersion))
	telemetry.Inc(fmt.Sprintf("go/platform/host/windows/build:%d", v.BuildNumber))
}
