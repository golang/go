// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !cmd_go_bootstrap && !unix && !windows

package telemetrystats

import "cmd/internal/telemetry/counter"

func incrementVersionCounters() {
	counter.Inc("go/platform:version-not-supported")
}
