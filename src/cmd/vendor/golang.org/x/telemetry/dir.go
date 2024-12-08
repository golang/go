// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package telemetry

import "golang.org/x/telemetry/internal/telemetry"

// Dir returns the telemetry directory.
func Dir() string {
	return telemetry.Default.Dir()
}
