// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go test cmd/go -v -run=^TestDocsUpToDate$ -fixdocs
//go:generate go test cmd/go -v -run=^TestCounterNamesUpToDate$ -update

package main

import (
	"cmd/internal/telemetry"
	"context"
	"flag"
	"fmt"
	"internal/buildcfg"
	"log"
	"os"
	"path/filepath"
	rtrace "runtime/trace"
	"slices"
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
	"cmd/go/internal/help"
	"cmd/go/internal/list"
	"cmd/go/internal/modcmd"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modget"
	"cmd/go/internal/modload"
	"cmd/go/internal/run"
	"cmd/go/internal/test"
	"cmd/go/internal/tool"
	"cmd/go/internal/toolchain"
	"cmd/go/internal/trace"
	"cmd/go/internal/version"
	"cmd/go/internal/vet"
	"cmd/go/internal/work"
	"cmd/go/internal/workcmd"
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
		modget.CmdGet,
		work.CmdInstall,
		list.CmdList,
		modcmd.CmdMod,
		workcmd.CmdWork,
		run.CmdRun,
		test.CmdTest,
		tool.CmdTool,
		version.CmdVersion,
		vet.CmdVet,

		help.HelpBuildConstraint,
		help.HelpBuildmode,
		help.HelpC,
		help.HelpCache,
		help.HelpEnvironment,
		help.HelpFileType,
		modload.HelpGoMod,
		help.HelpGopath,
		modfetch.HelpGoproxy,
		help.HelpImportPath,
		modload.HelpModules,
		modfetch.HelpModuleAuth,
		help.HelpPackages,
		modfetch.HelpPrivate,
		test.HelpTestflag,
		test.HelpTestfunc,
		modget.HelpVCS,
	}
}

var _ = go11tag

var counterErrorsGOPATHEntryRelative = base.NewCounter("go/errors:gopath-entry-relative")

func main() {
	log.SetFlags(0)
	telemetry.StartWithUpload() // Open the telemetry counter file so counters can be written to it.
	handleChdirFlag()
	toolchain.Select()

	flag.Usage = base.Usage
	flag.Parse()
	telemetry.CountFlags("go/flag:", *flag.CommandLine)

	args := flag.Args()
	if len(args) < 1 {
		base.Usage()
	}

	cfg.CmdName = args[0] // for error messages
	if args[0] == "help" {
		telemetry.Inc("go/subcommand:" + strings.Join(append([]string{"help"}, args[1:]...), "-"))
		help.Help(os.Stdout, args[1:])
		return
	}

	if cfg.GOROOT == "" {
		fmt.Fprintf(os.Stderr, "go: cannot find GOROOT directory: 'go' binary is trimmed and GOROOT is not set\n")
		os.Exit(2)
	}
	if fi, err := os.Stat(cfg.GOROOT); err != nil || !fi.IsDir() {
		fmt.Fprintf(os.Stderr, "go: cannot find GOROOT directory: %v\n", cfg.GOROOT)
		os.Exit(2)
	}

	// Diagnose common mistake: GOPATH==GOROOT.
	// This setting is equivalent to not setting GOPATH at all,
	// which is not what most people want when they do it.
	if gopath := cfg.BuildContext.GOPATH; filepath.Clean(gopath) == filepath.Clean(cfg.GOROOT) {
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
				if cfg.Getenv("GOPATH") == "" {
					// We inferred $GOPATH from $HOME and did a bad job at it.
					// Instead of dying, uninfer it.
					cfg.BuildContext.GOPATH = ""
				} else {
					counterErrorsGOPATHEntryRelative.Inc()
					fmt.Fprintf(os.Stderr, "go: GOPATH entry is relative; must be absolute path: %q.\nFor more details see: 'go help gopath'\n", p)
					os.Exit(2)
				}
			}
		}
	}

	cmd, used := lookupCmd(args)
	cfg.CmdName = strings.Join(args[:used], " ")
	if len(cmd.Commands) > 0 {
		if used >= len(args) {
			help.PrintUsage(os.Stderr, cmd)
			base.SetExitStatus(2)
			base.Exit()
		}
		if args[used] == "help" {
			// Accept 'go mod help' and 'go mod help foo' for 'go help mod' and 'go help mod foo'.
			telemetry.Inc("go/subcommand:" + strings.ReplaceAll(cfg.CmdName, " ", "-") + "-" + strings.Join(args[used:], "-"))
			help.Help(os.Stdout, append(slices.Clip(args[:used]), args[used+1:]...))
			base.Exit()
		}
		helpArg := ""
		if used > 0 {
			helpArg += " " + strings.Join(args[:used], " ")
		}
		cmdName := cfg.CmdName
		if cmdName == "" {
			cmdName = args[0]
		}
		telemetry.Inc("go/subcommand:unknown")
		fmt.Fprintf(os.Stderr, "go %s: unknown command\nRun 'go help%s' for usage.\n", cmdName, helpArg)
		base.SetExitStatus(2)
		base.Exit()
	}
	telemetry.Inc("go/subcommand:" + strings.ReplaceAll(cfg.CmdName, " ", "-"))
	invoke(cmd, args[used-1:])
	base.Exit()
}

// lookupCmd interprets the initial elements of args
// to find a command to run (cmd.Runnable() == true)
// or else a command group that ran out of arguments
// or had an unknown subcommand (len(cmd.Commands) > 0).
// It returns that command and the number of elements of args
// that it took to arrive at that command.
func lookupCmd(args []string) (cmd *base.Command, used int) {
	cmd = base.Go
	for used < len(args) {
		c := cmd.Lookup(args[used])
		if c == nil {
			break
		}
		if c.Runnable() {
			cmd = c
			used++
			break
		}
		if len(c.Commands) > 0 {
			cmd = c
			used++
			if used >= len(args) || args[0] == "help" {
				break
			}
			continue
		}
		// len(c.Commands) == 0 && !c.Runnable() => help text; stop at "help"
		break
	}
	return cmd, used
}

func invoke(cmd *base.Command, args []string) {
	// 'go env' handles checking the build config
	if cmd != envcmd.CmdEnv {
		buildcfg.Check()
		if cfg.ExperimentErr != nil {
			base.Fatal(cfg.ExperimentErr)
		}
	}

	// Set environment (GOOS, GOARCH, etc) explicitly.
	// In theory all the commands we invoke should have
	// the same default computation of these as we do,
	// but in practice there might be skew
	// This makes sure we all agree.
	cfg.OrigEnv = toolchain.FilterEnv(os.Environ())
	cfg.CmdEnv = envcmd.MkEnv()
	for _, env := range cfg.CmdEnv {
		if os.Getenv(env.Name) != env.Value {
			os.Setenv(env.Name, env.Value)
		}
	}

	cmd.Flag.Usage = func() { cmd.Usage() }
	if cmd.CustomFlags {
		args = args[1:]
	} else {
		base.SetFromGOFLAGS(&cmd.Flag)
		cmd.Flag.Parse(args[1:])
		telemetry.CountFlags("go/flag:"+strings.ReplaceAll(cfg.CmdName, " ", "-")+"-", cmd.Flag)
		args = cmd.Flag.Args()
	}

	if cfg.DebugRuntimeTrace != "" {
		f, err := os.Create(cfg.DebugRuntimeTrace)
		if err != nil {
			base.Fatalf("creating trace file: %v", err)
		}
		if err := rtrace.Start(f); err != nil {
			base.Fatalf("starting event trace: %v", err)
		}
		defer func() {
			rtrace.Stop()
		}()
	}

	ctx := maybeStartTrace(context.Background())
	ctx, span := trace.StartSpan(ctx, fmt.Sprint("Running ", cmd.Name(), " command"))
	cmd.Run(ctx, cmd, args)
	span.Done()
}

func init() {
	base.Usage = mainUsage
}

func mainUsage() {
	help.PrintUsage(os.Stderr, base.Go)
	os.Exit(2)
}

func maybeStartTrace(pctx context.Context) context.Context {
	if cfg.DebugTrace == "" {
		return pctx
	}

	ctx, close, err := trace.Start(pctx, cfg.DebugTrace)
	if err != nil {
		base.Fatalf("failed to start trace: %v", err)
	}
	base.AtExit(func() {
		if err := close(); err != nil {
			base.Fatalf("failed to stop trace: %v", err)
		}
	})

	return ctx
}

// handleChdirFlag handles the -C flag before doing anything else.
// The -C flag must be the first flag on the command line, to make it easy to find
// even with commands that have custom flag parsing.
// handleChdirFlag handles the flag by chdir'ing to the directory
// and then removing that flag from the command line entirely.
//
// We have to handle the -C flag this way for two reasons:
//
//  1. Toolchain selection needs to be in the right directory to look for go.mod and go.work.
//
//  2. A toolchain switch later on reinvokes the new go command with the same arguments.
//     The parent toolchain has already done the chdir; the child must not try to do it again.
func handleChdirFlag() {
	_, used := lookupCmd(os.Args[1:])
	used++ // because of [1:]
	if used >= len(os.Args) {
		return
	}

	var dir string
	switch a := os.Args[used]; {
	default:
		return

	case a == "-C", a == "--C":
		if used+1 >= len(os.Args) {
			return
		}
		dir = os.Args[used+1]
		os.Args = slices.Delete(os.Args, used, used+2)

	case strings.HasPrefix(a, "-C="), strings.HasPrefix(a, "--C="):
		_, dir, _ = strings.Cut(a, "=")
		os.Args = slices.Delete(os.Args, used, used+1)
	}
	telemetry.Inc("go/flag:C")

	if err := os.Chdir(dir); err != nil {
		base.Fatalf("go: %v", err)
	}
}
