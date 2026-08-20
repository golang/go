// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sgutil

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type PathList struct {
	paths     []string
	fromShell bool // if false, perform our own shell var expansion
}

func FlagPathList(name string, usage string, value ...string) *PathList {
	p := &PathList{paths: value, fromShell: false}
	flag.Var(p, name, usage)
	return p
}

func (l *PathList) String() string {
	return strings.Join(l.paths, string(filepath.ListSeparator))
}

func (l *PathList) Set(val string) error {
	l.paths = filepath.SplitList(val)
	l.fromShell = true
	return nil
}

// Find returns the first element of l containing a file by the given name. If
// file is not found in the path list, it returns a descriptive error.
func (l *PathList) Find(file string) (string, error) {
	for _, path := range l.paths {
		if !l.fromShell {
			path = os.ExpandEnv(path)
		}
		if path == "" {
			// Probably an unknown shell variable. Ignore.
			continue
		}
		if _, err := os.Stat(filepath.Join(path, file)); err == nil {
			return filepath.Abs(path)
		}
	}
	var errMsg strings.Builder
	fmt.Fprintf(&errMsg, "%q not found in any of:", file)
	for _, path := range l.paths {
		errMsg.WriteString("\n\t")
		errMsg.WriteString(path)
	}
	return "", fmt.Errorf("%s", errMsg.String())
}

// FlagXEDPath registers and returns a global -xedPath flag.
func FlagXEDPath(genRoot string) *PathList {
	return FlagPathList("xedPath", "`list` of directories to search for XED data (must be an XED obj/dgen directory)",
		"$XEDPATH", genRoot+"/extern/xed/obj/dgen", "$HOME/xed/obj/dgen")
}

func ResolveXEDPath(flag *PathList) (string, error) {
	xedPath, err := flag.Find("all-dec-instructions.txt")
	if err != nil {
		return "", fmt.Errorf(`%s
Could not find XED data. Use fetch-xed.sh to download it, or set
$XEDPATH or -xedPath to the XED obj/dgen directory.`, err)
	}
	return xedPath, nil
}

// FlagARM64Path registers and returns a global -arm64Path flag.
func FlagARM64Path(genRoot string) *PathList {
	const isaDir = "ISA_A64_xml_A_profile_2026-03_96-2026-03_rel"
	return FlagPathList("arm64Path", "`list` of directories to search for ARM64 ISA data",
		"$ARM64_ISA_PATH", genRoot+"/extern/"+isaDir, "$HOME/Downloads/"+isaDir)
}

func ResolveARM64Path(flag *PathList) (string, error) {
	arm64Path, err := flag.Find("abs_advsimd.xml")
	if err != nil {
		return "", fmt.Errorf(`%s
Could not find ARM64 ISA data. Use fetch-arm64.sh to download it, or set
$ARM64_ISA_PATH or -arm64Path to the ARM64 ISA specification directory.`, err)
	}
	return arm64Path, nil

}
