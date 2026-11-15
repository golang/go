// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/contextcmd"
	"cmd/go/internal/logos/eject"
	"cmd/go/internal/logos/lintcmd"
	"cmd/go/internal/logos/lsp"
	"cmd/go/internal/logos/next"
	"cmd/go/internal/logos/settings"
)

func ConfigureCommand(cmd *base.Command) {
	cmd.UsageLine = "logos"
	cmd.Long = "Logos is a tool for managing loGo/s source code."
	dropFmt(cmd)
	for _, sub := range cmd.Commands {
		rebrandUsageLine(sub)
	}
	contextcmd.Register(cmd)
	eject.Register(cmd)
	lintcmd.Register(cmd)
	lsp.Register(cmd)
	next.Register(cmd)
	addPedanticFlag(cmd.Lookup("build"))
	addPedanticFlag(cmd.Lookup("lint"))
	addJSONDiagnosticsFlag(cmd.Lookup("build"))
	addJSONDiagnosticsFlag(cmd.Lookup("lint"))
	addWasmLoaderHelp(cmd.Lookup("build"))
}

func addPedanticFlag(cmd *base.Command) {
	if cmd == nil {
		return
	}
	if cmd.Flag.Lookup("pedantic") == nil {
		cmd.Flag.Var(settings.PedanticFlag(&settings.CommandLine), "pedantic", "")
	}
	addHelp(cmd, "-pedantic", "\n\t-pedantic\n\t\tenable pedantic diagnostics (logos builds only).\n")
}

func addJSONDiagnosticsFlag(cmd *base.Command) {
	if cmd == nil {
		return
	}
	if cmd.Flag.Lookup("json-diagnostics") == nil {
		cmd.Flag.BoolVar(&settings.CommandLine.JSON, "json-diagnostics", false, "")
	}
	addHelp(cmd, "-json-diagnostics", "\n\t-json-diagnostics\n\t\tprint each compiler and lint diagnostic as one JSON object\n\t\tper line on stderr (logos builds only).\n")
}

func addWasmLoaderHelp(cmd *base.Command) {
	if cmd == nil {
		return
	}
	addHelp(cmd, "-wasm-loader", "\n\t-wasm-loader\n\t\twrite <name>.loader.mjs, the wasm host loader, beside the output of a\n\t\tGOARCH=wasm main package when -o names a file with a .wasm suffix.\n\t\tThe default is true; -wasm-loader=false turns it off.\n")
}

func addHelp(cmd *base.Command, flag, help string) {
	if strings.Contains(cmd.Long, flag) {
		return
	}
	before, after, found := strings.Cut(cmd.Long, "\nSee also:")
	if found {
		cmd.Long = before + help + "\nSee also:" + after
		return
	}
	cmd.Long += help
}

func rebrandUsageLine(cmd *base.Command) {
	if rest, ok := strings.CutPrefix(cmd.UsageLine, "go "); ok {
		cmd.UsageLine = "logos " + rest
	}
	for _, sub := range cmd.Commands {
		rebrandUsageLine(sub)
	}
}

func ProgramName() string {
	if cfg.Logos {
		return "logos"
	}
	return "go"
}
