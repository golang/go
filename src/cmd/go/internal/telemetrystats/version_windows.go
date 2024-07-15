// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !cmd_go_bootstrap && windows

package telemetrystats

import (
	"fmt"

	"cmd/internal/telemetry/counter"

	"golang.org/x/sys/windows"
)

func incrementVersionCounters() {
	v := windows.RtlGetVersion()
	counter.Inc(fmt.Sprintf("go/platform/host/windows/major-version:%d", v.MajorVersion))
	counter.Inc(fmt.Sprintf("go/platform/host/windows/version:%d-%d", v.MajorVersion, v.MinorVersion))
	counter.Inc(fmt.Sprintf("go/platform/host/windows/build:%d", v.BuildNumber))
}
