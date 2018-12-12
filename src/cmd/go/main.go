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
	"cmd/go/internal/modcmd"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modget"
	"cmd/go/internal/modload"
	"cmd/go/internal/run"
	"cmd/go/internal/test"
	"cmd/go/internal/tool"
	"cmd/go/internal/version"
	"cmd/go/internal/vet"
	"cmd/go/internal/work"
)

func init() {
	base.Go.Commands = []*base.Command{
		bug.CmdBug,
		work.CmdBuild,
		clean.CmdClean,
		doc.CmdDoc,
		envcmd.CmdEnv,
		fix.CmdFix,
		fmtcmd.CmdFmt,
		generate.CmdGenerate,
		get.CmdGet,
		work.CmdInstall,
		list.CmdList,
		modcmd.CmdMod,
		run.CmdRun,
		test.CmdTest,
		tool.CmdTool,
		version.CmdVersion,
		vet.CmdVet,

		help.HelpBuildmode,
		help.HelpC,
		help.HelpCache,
		help.HelpEnvironment,
		help.HelpFileType,
		modload.HelpGoMod,
		help.HelpGopath,
		get.HelpGopathGet,
		modfetch.HelpGoproxy,
		help.HelpImportPath,
		modload.HelpModules,
		modget.HelpModuleGet,
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

	if modload.MustUseModules {
		// If running with modules force-enabled, change get now to change help message.
		*get.CmdGet = *modget.CmdGet
	}

	if args[0] == "get" || args[0] == "help" {
		// Replace get with module-aware get if appropriate.
		// Note that if MustUseModules is true, this happened already above,
		// but no harm in doing it again.
		if modload.Init(); modload.Enabled() {
			*get.CmdGet = *modget.CmdGet
		}
	}

	cfg.CmdName = args[0] // for error messages
	if args[0] == "help" {
		help.Help(os.Stdout, args[1:])
		return
	}

	// Diagnose common mistake: GOPATH==GOROOT.
	// This setting is equivalent to not setting GOPATH at all,
	// which is not what most people want when they do it.
	if gopath := cfg.BuildContext.GOPATH; filepath.Clean(gopath) == filepath.Clean(runtime.GOROOT()) {
		fmt.Fprintf(os.Stderr, "warning: GOPATH set to GOROOT (%s) has no effect\n", gopath)
	} else {
		for _, p := range filepath.SplitList(gopath) {
			// Some GOPATHs have empty directory elements - ignore them.
			// See issue 21928 for details.
			if p == "" {
				continue
			}
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

	// TODO(rsc): Remove all these helper prints in Go 1.12.
	switch args[0] {
	case "mod":
		if len(args) >= 2 {
			flag := args[1]
			if strings.HasPrefix(flag, "--") {
				flag = flag[1:]
			}
			if i := strings.Index(flag, "="); i >= 0 {
				flag = flag[:i]
			}
			switch flag {
			case "-sync", "-fix":
				fmt.Fprintf(os.Stderr, "go: go mod %s is now go mod tidy\n", flag)
				os.Exit(2)
			case "-init", "-graph", "-vendor", "-verify":
				fmt.Fprintf(os.Stderr, "go: go mod %s is now go mod %s\n", flag, flag[1:])
				os.Exit(2)
			case "-fmt", "-json", "-module", "-require", "-droprequire", "-replace", "-dropreplace", "-exclude", "-dropexclude":
				fmt.Fprintf(os.Stderr, "go: go mod %s is now go mod edit %s\n", flag, flag)
				os.Exit(2)
			}
		}
	case "vendor":
		fmt.Fprintf(os.Stderr, "go: vgo vendor is now go mod vendor\n")
		os.Exit(2)
	case "verify":
		fmt.Fprintf(os.Stderr, "go: vgo verify is now go mod verify\n")
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

BigCmdLoop:
	for bigCmd := base.Go; ; {
		for _, cmd := range bigCmd.Commands {
			if cmd.Name() != args[0] {
				continue
			}
			if len(cmd.Commands) > 0 {
				bigCmd = cmd
				args = args[1:]
				if len(args) == 0 {
					help.PrintUsage(os.Stderr, bigCmd)
					base.SetExitStatus(2)
					base.Exit()
				}
				if args[0] == "help" {
					// Accept 'go mod help' and 'go mod help foo' for 'go help mod' and 'go help mod foo'.
					help.Help(os.Stdout, append(strings.Split(cfg.CmdName, " "), args[1:]...))
					return
				}
				cfg.CmdName += " " + args[0]
				continue BigCmdLoop
			}
			if !cmd.Runnable() {
				continue
			}
			cmd.Flag.Usage = func() { cmd.Usage() }
			if cmd.CustomFlags {
				args = args[1:]
			} else {
				base.SetFromGOFLAGS(cmd.Flag)
				cmd.Flag.Parse(args[1:])
				args = cmd.Flag.Args()
			}
			cmd.Run(cmd, args)
			base.Exit()
			return
		}
		helpArg := ""
		if i := strings.LastIndex(cfg.CmdName, " "); i >= 0 {
			helpArg = " " + cfg.CmdName[:i]
		}
		fmt.Fprintf(os.Stderr, "go %s: unknown command\nRun 'go help%s' for usage.\n", cfg.CmdName, helpArg)
		base.SetExitStatus(2)
		base.Exit()
	}
}

func init() {
	base.Usage = mainUsage
}

func mainUsage() {
	help.PrintUsage(os.Stderr, base.Go)
	os.Exit(2)
}
