// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package telemetrycmd implements the "go telemetry" command.
package telemetrycmd

import (
	"context"
	"fmt"
	"os"

	"cmd/go/internal/base"
	"cmd/internal/telemetry"
)

var CmdTelemetry = &base.Command{
	UsageLine: "go telemetry [off|local|on]",
	Short:     "manage telemetry data and settings",
	Long: `Telemetry is used to manage Go telemetry data and settings.

Telemetry can be in one of three modes: off, local, or on.

When telemetry is in local mode, counter data is written to the local file
system, but will not be uploaded to remote servers.

When telemetry is off, local counter data is neither collected nor uploaded.

When telemetry is on, telemetry data is written to the local file system
and periodically sent to https://telemetry.go.dev/. Uploaded data is used to
help improve the Go toolchain and related tools, and it will be published as
part of a public dataset.

For more details, see https://telemetry.go.dev/privacy.
This data is collected in accordance with the Google Privacy Policy
(https://policies.google.com/privacy).

To view the current telemetry mode, run "go telemetry".
To disable telemetry uploading, but keep local data collection, run
"go telemetry local".
To enable both collection and uploading, run “go telemetry on”.
To disable both collection and uploading, run "go telemetry off".

The current telemetry mode is also available as the value of the
non-settable "GOTELEMETRY" go env variable. The directory in the
local file system that telemetry data is written to is available
as the value of the non-settable "GOTELEMETRYDIR" go env variable.

See https://go.dev/doc/telemetry for more information on telemetry.
`,
	Run: runTelemetry,
}

func init() {
	base.AddChdirFlag(&CmdTelemetry.Flag)
}

func runTelemetry(ctx context.Context, cmd *base.Command, args []string) {
	if len(args) == 0 {
		fmt.Println(telemetry.Mode())
		return
	}

	if len(args) != 1 {
		cmd.Usage()
	}

	mode := args[0]
	if mode != "local" && mode != "off" && mode != "on" {
		cmd.Usage()
	}
	if old := telemetry.Mode(); old == mode {
		return
	}

	if err := telemetry.SetMode(mode); err != nil {
		base.Fatalf("go: failed to set the telemetry mode to %s: %v", mode, err)
	}
	if mode == "on" {
		fmt.Fprintln(os.Stderr, telemetryOnMessage())
	}
}

func telemetryOnMessage() string {
	return `Telemetry uploading is now enabled and data will be periodically sent to
https://telemetry.go.dev/. Uploaded data is used to help improve the Go
toolchain and related tools, and it will be published as part of a public
dataset.

For more details, see https://telemetry.go.dev/privacy.
This data is collected in accordance with the Google Privacy Policy
(https://policies.google.com/privacy).

To disable telemetry uploading, but keep local data collection, run
“go telemetry local”.
To disable both collection and uploading, run “go telemetry off“.`
}
