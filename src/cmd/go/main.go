// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"go/build"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"text/template"
)

// A Command is an implementation of a go command
// like go build or go fix.
type Command struct {
	// Run runs the command.
	// The args are the arguments after the command name.
	Run func(cmd *Command, args []string)

	// UsageLine is the one-line usage message.
	// The first word in the line is taken to be the command name.
	UsageLine string

	// Short is the short description shown in the 'go help' output.
	Short string

	// Long is the long message shown in the 'go help <this-command>' output.
	Long string

	// Flag is a set of flags specific to this command.
	Flag flag.FlagSet

	// CustomFlags indicates that the command will do its own
	// flag parsing.
	CustomFlags bool
}

// Name returns the command's name: the first word in the usage line.
func (c *Command) Name() string {
	name := c.UsageLine
	i := strings.Index(name, " ")
	if i >= 0 {
		name = name[:i]
	}
	return name
}

func (c *Command) Usage() {
	fmt.Fprintf(os.Stderr, "usage: %s\n\n", c.UsageLine)
	fmt.Fprintf(os.Stderr, "%s\n", strings.TrimSpace(c.Long))
	os.Exit(2)
}

// Commands lists the available commands and help topics.
// The order here is the order in which they are printed by 'go help'.
var commands = []*Command{
	cmdBuild,
	cmdDoc,
	cmdFix,
	cmdFmt,
	cmdGet,
	cmdInstall,
	cmdList,
	cmdRun,
	cmdTest,
	cmdVersion,
	cmdVet,

	helpGopath,
	helpImportpath,
	helpRemote,
	helpTestflag,
	helpTestfunc,
}

var exitStatus = 0

func main() {
	flag.Usage = usage
	flag.Parse()
	log.SetFlags(0)

	args := flag.Args()
	if len(args) < 1 {
		usage()
	}

	if args[0] == "help" {
		help(args[1:])
		return
	}

	for _, cmd := range commands {
		if cmd.Name() == args[0] && cmd.Run != nil {
			cmd.Flag.Usage = func() { cmd.Usage() }
			if cmd.CustomFlags {
				args = args[1:]
			} else {
				cmd.Flag.Parse(args[1:])
				args = cmd.Flag.Args()
			}
			cmd.Run(cmd, args)
			exit()
			return
		}
	}

	fmt.Fprintf(os.Stderr, "Unknown command %#q\n\n", args[0])
	usage()
}

var usageTemplate = `usage: go command [arguments]

go manages Go source code.

The commands are:
{{range .}}{{if .Run}}
    {{.Name | printf "%-11s"}} {{.Short}}{{end}}{{end}}

Use "go help [command]" for more information about a command.

Additional help topics:
{{range .}}{{if not .Run}}
    {{.Name | printf "%-11s"}} {{.Short}}{{end}}{{end}}

Use "go help [topic]" for more information about that topic.

`

var helpTemplate = `{{if .Run}}usage: go {{.UsageLine}}

{{end}}{{.Long | trim}}
`

// tmpl executes the given template text on data, writing the result to w.
func tmpl(w io.Writer, text string, data interface{}) {
	t := template.New("top")
	t.Funcs(template.FuncMap{"trim": strings.TrimSpace})
	template.Must(t.Parse(text))
	if err := t.Execute(w, data); err != nil {
		panic(err)
	}
}

func printUsage(w io.Writer) {
	tmpl(w, usageTemplate, commands)
}

func usage() {
	printUsage(os.Stderr)
	os.Exit(2)
}

// help implements the 'help' command.
func help(args []string) {
	if len(args) == 0 {
		printUsage(os.Stdout)
		// not exit 2: succeeded at 'go help'.
		return
	}
	if len(args) != 1 {
		fmt.Fprintf(os.Stderr, "usage: go help command\n\nToo many arguments given.\n")
		os.Exit(2) // failed at 'go help'
	}

	arg := args[0]
	for _, cmd := range commands {
		if cmd.Name() == arg {
			tmpl(os.Stdout, helpTemplate, cmd)
			// not exit 2: succeeded at 'go help cmd'.
			return
		}
	}

	fmt.Fprintf(os.Stderr, "Unknown help topic %#q.  Run 'go help'.\n", arg)
	os.Exit(2) // failed at 'go help cmd'
}

// importPaths returns the import paths to use for the given command line.
func importPaths(args []string) []string {
	if len(args) == 1 {
		if args[0] == "all" || args[0] == "std" {
			return allPackages(args[0])
		}
	}
	if len(args) == 0 {
		return []string{"."}
	}
	return args
}

var atexitFuncs []func()

func atexit(f func()) {
	atexitFuncs = append(atexitFuncs, f)
}

func exit() {
	for _, f := range atexitFuncs {
		f()
	}
	os.Exit(exitStatus)
}

func fatalf(format string, args ...interface{}) {
	errorf(format, args...)
	exit()
}

func errorf(format string, args ...interface{}) {
	log.Printf(format, args...)
	exitStatus = 1
}

var logf = log.Printf

func exitIfErrors() {
	if exitStatus != 0 {
		exit()
	}
}

func run(cmdline ...string) {
	cmd := exec.Command(cmdline[0], cmdline[1:]...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		errorf("%v", err)
	}
}

// allPackages returns all the packages that can be found
// under the $GOPATH directories and $GOROOT.
func allPackages(what string) []string {
	have := map[string]bool{
		"builtin": true, // ignore pseudo-package that exists only for documentation
	}
	/*
		if !build.DefaultContext.CgoEnabled {
			have["runtime/cgo"] = true // ignore during walk
		}
	*/
	var pkgs []string

	// Commands
	goroot := build.Path[0].Path
	cmd := filepath.Join(goroot, "src/cmd") + string(filepath.Separator)
	filepath.Walk(cmd, func(path string, fi os.FileInfo, err error) error {
		if err != nil || !fi.IsDir() {
			return nil
		}
		name := path[len(cmd):]
		// Commands are all in cmd/, not in subdirectories.
		if strings.Contains(name, string(filepath.Separator)) {
			return filepath.SkipDir
		}

		_, err = build.ScanDir(path)
		if err != nil {
			return nil
		}

		// We use, e.g., cmd/gofmt as the pseudo import path for gofmt.
		name = "cmd/" + name
		if !have[name] {
			have[name] = true
			pkgs = append(pkgs, name)
		}
		return nil
	})

	for _, t := range build.Path {
		if what == "std" && !t.Goroot {
			continue
		}
		src := t.SrcDir() + string(filepath.Separator)
		filepath.Walk(src, func(path string, fi os.FileInfo, err error) error {
			if err != nil || !fi.IsDir() {
				return nil
			}

			// Avoid testdata directory trees.
			if strings.HasSuffix(path, string(filepath.Separator)+"testdata") {
				return filepath.SkipDir
			}

			name := filepath.ToSlash(path[len(src):])
			if what == "std" && strings.Contains(name, ".") {
				return filepath.SkipDir
			}
			if have[name] {
				return nil
			}

			_, err = build.ScanDir(path)
			if err != nil {
				return nil
			}

			pkgs = append(pkgs, name)
			have[name] = true

			// Avoid go/build test data.
			// TODO: Move it into a testdata directory.
			if path == filepath.Join(build.Path[0].SrcDir(), "go/build") {
				return filepath.SkipDir
			}

			return nil
		})
	}
	return pkgs
}
