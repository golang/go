// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package version implements the “go version” command.
package version

import (
	"context"
	"debug/buildinfo"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"cmd/go/internal/base"
)

var CmdVersion = &base.Command{
	UsageLine: "go version [-m] [-v] [file ...]",
	Short:     "print Go version",
	Long: `Version prints the build information for Go executables.

Go version reports the Go version used to build each of the named
executable files.

If no files are named on the command line, go version prints its own
version information.

If a directory is named, go version walks that directory, recursively,
looking for recognized Go binaries and reporting their versions.
By default, go version does not report unrecognized files found
during a directory scan. The -v flag causes it to report unrecognized files.

The -m flag causes go version to print each executable's embedded
module version information, when available. In the output, the module
information consists of multiple lines following the version line, each
indented by a leading tab character.

See also: go doc runtime/debug.BuildInfo.
`,
}

func init() {
	CmdVersion.Run = runVersion // break init cycle
}

var (
	versionM = CmdVersion.Flag.Bool("m", false, "")
	versionV = CmdVersion.Flag.Bool("v", false, "")
)

func runVersion(ctx context.Context, cmd *base.Command, args []string) {
	if len(args) == 0 {
		// If any of this command's flags were passed explicitly, error
		// out, because they only make sense with arguments.
		//
		// Don't error if the flags came from GOFLAGS, since that can be
		// a reasonable use case. For example, imagine GOFLAGS=-v to
		// turn "verbose mode" on for all Go commands, which should not
		// break "go version".
		var argOnlyFlag string
		if !base.InGOFLAGS("-m") && *versionM {
			argOnlyFlag = "-m"
		} else if !base.InGOFLAGS("-v") && *versionV {
			argOnlyFlag = "-v"
		}
		if argOnlyFlag != "" {
			fmt.Fprintf(os.Stderr, "go: 'go version' only accepts %s flag with arguments\n", argOnlyFlag)
			base.SetExitStatus(2)
			return
		}
		fmt.Printf("go version %s %s/%s\n", runtime.Version(), runtime.GOOS, runtime.GOARCH)
		return
	}

	for _, arg := range args {
		info, err := os.Stat(arg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			base.SetExitStatus(1)
			continue
		}
		if info.IsDir() {
			scanDir(arg)
		} else {
			scanFile(arg, info, true)
		}
	}
}

// scanDir scans a directory for executables to run scanFile on.
func scanDir(dir string) {
	filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if d.Type().IsRegular() || d.Type()&fs.ModeSymlink != 0 {
			info, err := d.Info()
			if err != nil {
				if *versionV {
					fmt.Fprintf(os.Stderr, "%s: %v\n", path, err)
				}
				return nil
			}
			scanFile(path, info, *versionV)
		}
		return nil
	})
}

// isExe reports whether the file should be considered executable.
func isExe(file string, info fs.FileInfo) bool {
	if runtime.GOOS == "windows" {
		return strings.HasSuffix(strings.ToLower(file), ".exe")
	}
	return info.Mode().IsRegular() && info.Mode()&0111 != 0
}

// scanFile scans file to try to report the Go and module versions.
// If mustPrint is true, scanFile will report any error reading file.
// Otherwise (mustPrint is false, because scanFile is being called
// by scanDir) scanFile prints nothing for non-Go executables.
func scanFile(file string, info fs.FileInfo, mustPrint bool) {
	if info.Mode()&fs.ModeSymlink != 0 {
		// Accept file symlinks only.
		i, err := os.Stat(file)
		if err != nil || !i.Mode().IsRegular() {
			if mustPrint {
				fmt.Fprintf(os.Stderr, "%s: symlink\n", file)
			}
			return
		}
		info = i
	}

	if !isExe(file, info) {
		if mustPrint {
			fmt.Fprintf(os.Stderr, "%s: not executable file\n", file)
		}
		return
	}

	bi, err := buildinfo.ReadFile(file)
	if err != nil {
		if mustPrint {
			if pathErr := (*os.PathError)(nil); errors.As(err, &pathErr) && filepath.Clean(pathErr.Path) == filepath.Clean(file) {
				fmt.Fprintf(os.Stderr, "%v\n", file)
			} else {
				fmt.Fprintf(os.Stderr, "%s: %v\n", file, err)
			}
		}
		return
	}

	fmt.Printf("%s: %s\n", file, bi.GoVersion)
	bi.GoVersion = "" // suppress printing go version again
	mod := bi.String()
	if *versionM && len(mod) > 0 {
		fmt.Printf("\t%s\n", strings.ReplaceAll(mod[:len(mod)-1], "\n", "\n\t"))
	}
}
