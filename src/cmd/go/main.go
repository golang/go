// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go test cmd/go -v -run=^TestDocsUpToDate$ -fixdocs

package main

import (
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
	"cmd/go/internal/telemetrycmd"
	"cmd/go/internal/telemetrystats"
	"cmd/go/internal/test"
	"cmd/go/internal/tool"
	"cmd/go/internal/toolchain"
	"cmd/go/internal/trace"
	"cmd/go/internal/version"
	"cmd/go/internal/vet"
	"cmd/go/internal/work"
	"cmd/go/internal/workcmd"
	"cmd/internal/telemetry"
	"cmd/internal/telemetry/counter"
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
		telemetrycmd.CmdTelemetry,
		test.CmdTest,
		tool.CmdTool,
		version.CmdVersion,
		vet.CmdVet,

		help.HelpBuildConstraint,
		help.HelpBuildJSON,
		help.HelpBuildmode,
		help.HelpC,
		help.HelpCache,
		help.HelpEnvironment,
		help.HelpFileType,
		help.HelpGoAuth,
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

var counterErrorsGOPATHEntryRelative = counter.New("go/errors:gopath-entry-relative")

func main() {
	log.SetFlags(0)
	telemetry.MaybeChild() // Run in child mode if this is the telemetry sidecar child process.
	cmdIsGoTelemetryOff := cmdIsGoTelemetryOff()
	if !cmdIsGoTelemetryOff {
		counter.Open() // Open the telemetry counter file so counters can be written to it.
	}
	handleChdirFlag()
	toolchain.Select()

	if !cmdIsGoTelemetryOff {
		telemetry.MaybeParent() // Run the upload process. Opening the counter file is idempotent.
	}
	// Add global -help flag support
	var globalHelp bool
	flag.BoolVar(&globalHelp, "help", false, "show help")
	flag.Usage = base.Usage
	flag.Parse()
	counter.Inc("go/invocations")
	counter.CountFlags("go/flag:", *flag.CommandLine)

	args := flag.Args()

	// Handle global -help flag
	if globalHelp {
		help.Help(os.Stdout, nil)
		return
	}

	if len(args) < 1 {
		base.Usage()
	}

	cfg.CmdName = args[0] // for error messages
	if args[0] == "help" {
		counter.Inc("go/subcommand:" + strings.Join(append([]string{"help"}, args[1:]...), "-"))
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
	switch strings.ToLower(cfg.GOROOT) {
	case "/usr/local/go": // Location recommended for installation on Linux and Darwin and used by Mac installer.
		counter.Inc("go/goroot:usr-local-go")
	case "/usr/lib/go": // A typical location used by Linux package managers.
		counter.Inc("go/goroot:usr-lib-go")
	case "/usr/lib/golang": // Another typical location used by Linux package managers.
		counter.Inc("go/goroot:usr-lib-golang")
	case `c:\program files\go`: // Location used by Windows installer.
		counter.Inc("go/goroot:program-files-go")
	case `c:\program files (x86)\go`: // Location used by 386 Windows installer on amd64 platform.
		counter.Inc("go/goroot:program-files-x86-go")
	default:
		counter.Inc("go/goroot:other")
	}

	// Diagnose common mistake: GOPATH==GOROOT.
	// This setting is equivalent to not setting GOPATH at all,
	// which is not what most people want when they do it.
	if gopath := cfg.BuildContext.GOPATH; filepath.Clean(gopath) == filepath.Clean(cfg.GOROOT) {
		fmt.Fprintf(os.Stderr, "warning: both GOPATH and GOROOT are the same directory (%s); see https://go.dev/wiki/InstallTroubleshooting\n", gopath)
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
			counter.Inc("go/subcommand:" + strings.ReplaceAll(cfg.CmdName, " ", "-") + "-" + strings.Join(args[used:], "-"))
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
		counter.Inc("go/subcommand:unknown")
		fmt.Fprintf(os.Stderr, "go %s: unknown command\nRun 'go help%s' for usage.\n", cmdName, helpArg)
		base.SetExitStatus(2)
		base.Exit()
	}
	// Increment a subcommand counter for the subcommand we're running.
	// Don't increment the counter for the tool subcommand here: we'll
	// increment in the tool subcommand's Run function because we need
	// to do the flag processing in invoke first.
	if cfg.CmdName != "tool" {
		counter.Inc("go/subcommand:" + strings.ReplaceAll(cfg.CmdName, " ", "-"))
	}
	telemetrystats.Increment()
	invoke(cmd, args[used-1:])
	base.Exit()
}

// cmdIsGoTelemetryOff reports whether the command is "go telemetry off". This
// is used to decide whether to disable the opening of counter files. See #69269.
func cmdIsGoTelemetryOff() bool {
	restArgs := os.Args[1:]
	// skipChdirFlag skips the -C flag, which is the only flag that can appear
	// in a valid 'go telemetry off' command, and which hasn't been processed
	// yet. We need to determine if the command is 'go telemetry off' before we open
	// the counter file, but we want to process -C after we open counters so that
	// we can increment the flag counter for it.
	skipChdirFlag := func() {
		if len(restArgs) == 0 {
			return
		}
		switch a := restArgs[0]; {
		case a == "-C", a == "--C":
			if len(restArgs) < 2 {
				restArgs = nil
				return
			}
			restArgs = restArgs[2:]

		case strings.HasPrefix(a, "-C="), strings.HasPrefix(a, "--C="):
			restArgs = restArgs[1:]
		}
	}
	skipChdirFlag()
	cmd, used := lookupCmd(restArgs)
	if cmd != telemetrycmd.CmdTelemetry {
		return false
	}
	restArgs = restArgs[used:]
	skipChdirFlag()
	return len(restArgs) == 1 && restArgs[0] == "off"
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

	// Add -help flag support to all commands
	var helpFlag bool
	if !cmd.CustomFlags {
		cmd.Flag.BoolVar(&helpFlag, "help", false, "show help")
	}

	cmd.Flag.Usage = func() {
		if helpFlag {
			// Show full help like "go help <command>"
			help.Help(os.Stdout, strings.Fields(cmd.LongName()))
			base.Exit()
		} else {
			cmd.Usage()
		}
	}
	if cmd.CustomFlags {
		args = args[1:]
	} else {
		base.SetFromGOFLAGS(&cmd.Flag)
		cmd.Flag.Parse(args[1:])

		// Check if -help flag was set and show full help
		if helpFlag {
			help.Help(os.Stdout, strings.Fields(cmd.LongName()))
			base.Exit()
		}

		flagCounterPrefix := "go/" + strings.ReplaceAll(cfg.CmdName, " ", "-") + "/flag"
		counter.CountFlags(flagCounterPrefix+":", cmd.Flag)
		counter.CountFlagValue(flagCounterPrefix+"/", cmd.Flag, "buildmode")
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
			f.Close()
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
	counter.Inc("go/flag:C")

	if err := os.Chdir(dir); err != nil {
		base.Fatalf("go: %v", err)
	}
}
