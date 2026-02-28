// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !cmd_go_bootstrap && windows

package telemetrystats

import (
	"fmt"
	"internal/syscall/windows"

	"cmd/internal/telemetry/counter"
)

func incrementVersionCounters() {
	major, minor, build := windows.Version()
	counter.Inc(fmt.Sprintf("go/platform/host/windows/major-version:%d", major))
	counter.Inc(fmt.Sprintf("go/platform/host/windows/version:%d-%d", major, minor))
	counter.Inc(fmt.Sprintf("go/platform/host/windows/build:%d", build))
}
