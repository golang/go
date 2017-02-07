// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate ./mkalldocs.sh

package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/bug"
	"cmd/go/internal/cfg"
	"cmd/go/internal/clean"
	"cmd/go/internal/doc"
	"cmd/go/internal/envcmd"
	"cmd/go/internal/fix"
	"cmd/go/internal/fmtcmd"
	"cmd/go/internal/generate"
	"cmd/go/internal/get"
	"cmd/go/internal/help"
	"cmd/go/internal/list"
	"cmd/go/internal/run"
	"cmd/go/internal/test"
	"cmd/go/internal/tool"
	"cmd/go/internal/version"
	"cmd/go/internal/vet"
	"cmd/go/internal/work"
)

func init() {
	base.Commands = []*base.Command{
		work.CmdBuild,
		clean.CmdClean,
		doc.CmdDoc,
		envcmd.CmdEnv,
		bug.CmdBug,
		fix.CmdFix,
		fmtcmd.CmdFmt,
		generate.CmdGenerate,
		get.CmdGet,
		work.CmdInstall,
		list.CmdList,
		run.CmdRun,
		test.CmdTest,
		tool.CmdTool,
		version.CmdVersion,
		vet.CmdVet,

		help.HelpC,
		help.HelpBuildmode,
		help.HelpFileType,
		help.HelpGopath,
		help.HelpEnvironment,
		help.HelpImportPath,
		help.HelpPackages,
		test.HelpTestflag,
		test.HelpTestfunc,
	}
}

func main() {
	_ = go11tag
	flag.Usage = base.Usage
	flag.Parse()
	log.SetFlags(0)

	args := flag.Args()
	if len(args) < 1 {
		base.Usage()
	}

	if args[0] == "help" {
		help.Help(args[1:])
		return
	}

	// Diagnose common mistake: GOPATH==GOROOT.
	// This setting is equivalent to not setting GOPATH at all,
	// which is not what most people want when they do it.
	if gopath := cfg.BuildContext.GOPATH; filepath.Clean(gopath) == filepath.Clean(runtime.GOROOT()) {
		fmt.Fprintf(os.Stderr, "warning: GOPATH set to GOROOT (%s) has no effect\n", gopath)
	} else {
		for _, p := range filepath.SplitList(gopath) {
			// Note: using HasPrefix instead of Contains because a ~ can appear
			// in the middle of directory elements, such as /tmp/git-1.8.2~rc3
			// or C:\PROGRA~1. Only ~ as a path prefix has meaning to the shell.
			if strings.HasPrefix(p, "~") {
				fmt.Fprintf(os.Stderr, "go: GOPATH entry cannot start with shell metacharacter '~': %q\n", p)
				os.Exit(2)
			}
			if !filepath.IsAbs(p) {
				fmt.Fprintf(os.Stderr, "go: GOPATH entry is relative; must be absolute path: %q.\nFor more details see: 'go help gopath'\n", p)
				os.Exit(2)
			}
		}
	}

	if fi, err := os.Stat(cfg.GOROOT); err != nil || !fi.IsDir() {
		fmt.Fprintf(os.Stderr, "go: cannot find GOROOT directory: %v\n", cfg.GOROOT)
		os.Exit(2)
	}

	// Set environment (GOOS, GOARCH, etc) explicitly.
	// In theory all the commands we invoke should have
	// the same default computation of these as we do,
	// but in practice there might be skew
	// This makes sure we all agree.
	cfg.OrigEnv = os.Environ()
	cfg.CmdEnv = envcmd.MkEnv()
	for _, env := range cfg.CmdEnv {
		if os.Getenv(env.Name) != env.Value {
			os.Setenv(env.Name, env.Value)
		}
	}

	for _, cmd := range base.Commands {
		if cmd.Name() == args[0] && cmd.Runnable() {
			cmd.Flag.Usage = func() { cmd.Usage() }
			if cmd.CustomFlags {
				args = args[1:]
			} else {
				cmd.Flag.Parse(args[1:])
				args = cmd.Flag.Args()
			}
			cmd.Run(cmd, args)
			base.Exit()
			return
		}
	}

	fmt.Fprintf(os.Stderr, "go: unknown subcommand %q\nRun 'go help' for usage.\n", args[0])
	base.SetExitStatus(2)
	base.Exit()
}

func init() {
	base.Usage = mainUsage
}

func mainUsage() {
	// special case "go test -h"
	if len(os.Args) > 1 && os.Args[1] == "test" {
		test.Usage()
	}
	help.PrintUsage(os.Stderr)
	os.Exit(2)
}
