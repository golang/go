// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package driver

import (
	"fmt"
	"os"
	"strings"

	"github.com/google/pprof/internal/binutils"
	"github.com/google/pprof/internal/plugin"
)

type source struct {
	Sources  []string
	ExecName string
	BuildID  string
	Base     []string

	Seconds   int
	Timeout   int
	Symbolize string
}

// Parse parses the command lines through the specified flags package
// and returns the source of the profile and optionally the command
// for the kind of report to generate (nil for interactive use).
func parseFlags(o *plugin.Options) (*source, []string, error) {
	flag := o.Flagset
	// Comparisons.
	flagBase := flag.StringList("base", "", "Source for base profile for comparison")
	// Internal options.
	flagSymbolize := flag.String("symbolize", "", "Options for profile symbolization")
	flagBuildID := flag.String("buildid", "", "Override build id for first mapping")
	// CPU profile options
	flagSeconds := flag.Int("seconds", -1, "Length of time for dynamic profiles")
	// Heap profile options
	flagInUseSpace := flag.Bool("inuse_space", false, "Display in-use memory size")
	flagInUseObjects := flag.Bool("inuse_objects", false, "Display in-use object counts")
	flagAllocSpace := flag.Bool("alloc_space", false, "Display allocated memory size")
	flagAllocObjects := flag.Bool("alloc_objects", false, "Display allocated object counts")
	// Contention profile options
	flagTotalDelay := flag.Bool("total_delay", false, "Display total delay at each region")
	flagContentions := flag.Bool("contentions", false, "Display number of delays at each region")
	flagMeanDelay := flag.Bool("mean_delay", false, "Display mean delay at each region")
	flagTools := flag.String("tools", os.Getenv("PPROF_TOOLS"), "Path for object tool pathnames")

	flagTimeout := flag.Int("timeout", -1, "Timeout in seconds for fetching a profile")

	// Flags used during command processing
	installedFlags := installFlags(flag)

	flagCommands := make(map[string]*bool)
	flagParamCommands := make(map[string]*string)
	for name, cmd := range pprofCommands {
		if cmd.hasParam {
			flagParamCommands[name] = flag.String(name, "", "Generate a report in "+name+" format, matching regexp")
		} else {
			flagCommands[name] = flag.Bool(name, false, "Generate a report in "+name+" format")
		}
	}

	args := flag.Parse(func() {
		o.UI.Print(usageMsgHdr +
			usage(true) +
			usageMsgSrc +
			flag.ExtraUsage() +
			usageMsgVars)
	})
	if len(args) == 0 {
		return nil, nil, fmt.Errorf("no profile source specified")
	}

	var execName string
	// Recognize first argument as an executable or buildid override.
	if len(args) > 1 {
		arg0 := args[0]
		if file, err := o.Obj.Open(arg0, 0, ^uint64(0), 0); err == nil {
			file.Close()
			execName = arg0
			args = args[1:]
		} else if *flagBuildID == "" && isBuildID(arg0) {
			*flagBuildID = arg0
			args = args[1:]
		}
	}

	// Report conflicting options
	if err := updateFlags(installedFlags); err != nil {
		return nil, nil, err
	}

	cmd, err := outputFormat(flagCommands, flagParamCommands)
	if err != nil {
		return nil, nil, err
	}

	si := pprofVariables["sample_index"].value
	si = sampleIndex(flagTotalDelay, si, "delay", "-total_delay", o.UI)
	si = sampleIndex(flagMeanDelay, si, "delay", "-mean_delay", o.UI)
	si = sampleIndex(flagContentions, si, "contentions", "-contentions", o.UI)
	si = sampleIndex(flagInUseSpace, si, "inuse_space", "-inuse_space", o.UI)
	si = sampleIndex(flagInUseObjects, si, "inuse_objects", "-inuse_objects", o.UI)
	si = sampleIndex(flagAllocSpace, si, "alloc_space", "-alloc_space", o.UI)
	si = sampleIndex(flagAllocObjects, si, "alloc_objects", "-alloc_objects", o.UI)
	pprofVariables.set("sample_index", si)

	if *flagMeanDelay {
		pprofVariables.set("mean", "true")
	}

	source := &source{
		Sources:   args,
		ExecName:  execName,
		BuildID:   *flagBuildID,
		Seconds:   *flagSeconds,
		Timeout:   *flagTimeout,
		Symbolize: *flagSymbolize,
	}

	for _, s := range *flagBase {
		if *s != "" {
			source.Base = append(source.Base, *s)
		}
	}

	if bu, ok := o.Obj.(*binutils.Binutils); ok {
		bu.SetTools(*flagTools)
	}
	return source, cmd, nil
}

// installFlags creates command line flags for pprof variables.
func installFlags(flag plugin.FlagSet) flagsInstalled {
	f := flagsInstalled{
		ints:    make(map[string]*int),
		bools:   make(map[string]*bool),
		floats:  make(map[string]*float64),
		strings: make(map[string]*string),
	}
	for n, v := range pprofVariables {
		switch v.kind {
		case boolKind:
			if v.group != "" {
				// Set all radio variables to false to identify conflicts.
				f.bools[n] = flag.Bool(n, false, v.help)
			} else {
				f.bools[n] = flag.Bool(n, v.boolValue(), v.help)
			}
		case intKind:
			f.ints[n] = flag.Int(n, v.intValue(), v.help)
		case floatKind:
			f.floats[n] = flag.Float64(n, v.floatValue(), v.help)
		case stringKind:
			f.strings[n] = flag.String(n, v.value, v.help)
		}
	}
	return f
}

// updateFlags updates the pprof variables according to the flags
// parsed in the command line.
func updateFlags(f flagsInstalled) error {
	vars := pprofVariables
	groups := map[string]string{}
	for n, v := range f.bools {
		vars.set(n, fmt.Sprint(*v))
		if *v {
			g := vars[n].group
			if g != "" && groups[g] != "" {
				return fmt.Errorf("conflicting options %q and %q set", n, groups[g])
			}
			groups[g] = n
		}
	}
	for n, v := range f.ints {
		vars.set(n, fmt.Sprint(*v))
	}
	for n, v := range f.floats {
		vars.set(n, fmt.Sprint(*v))
	}
	for n, v := range f.strings {
		vars.set(n, *v)
	}
	return nil
}

type flagsInstalled struct {
	ints    map[string]*int
	bools   map[string]*bool
	floats  map[string]*float64
	strings map[string]*string
}

// isBuildID determines if the profile may contain a build ID, by
// checking that it is a string of hex digits.
func isBuildID(id string) bool {
	return strings.Trim(id, "0123456789abcdefABCDEF") == ""
}

func sampleIndex(flag *bool, si string, sampleType, option string, ui plugin.UI) string {
	if *flag {
		if si == "" {
			return sampleType
		}
		ui.PrintErr("Multiple value selections, ignoring ", option)
	}
	return si
}

func outputFormat(bcmd map[string]*bool, acmd map[string]*string) (cmd []string, err error) {
	for n, b := range bcmd {
		if *b {
			if cmd != nil {
				return nil, fmt.Errorf("must set at most one output format")
			}
			cmd = []string{n}
		}
	}
	for n, s := range acmd {
		if *s != "" {
			if cmd != nil {
				return nil, fmt.Errorf("must set at most one output format")
			}
			cmd = []string{n, *s}
		}
	}
	return cmd, nil
}

var usageMsgHdr = "usage: pprof [options] [-base source] [binary] <source> ...\n"

var usageMsgSrc = "\n\n" +
	"  Source options:\n" +
	"    -seconds              Duration for time-based profile collection\n" +
	"    -timeout              Timeout in seconds for profile collection\n" +
	"    -buildid              Override build id for main binary\n" +
	"    -base source          Source of profile to use as baseline\n" +
	"    profile.pb.gz         Profile in compressed protobuf format\n" +
	"    legacy_profile        Profile in legacy pprof format\n" +
	"    http://host/profile   URL for profile handler to retrieve\n" +
	"    -symbolize=           Controls source of symbol information\n" +
	"      none                  Do not attempt symbolization\n" +
	"      local                 Examine only local binaries\n" +
	"      fastlocal             Only get function names from local binaries\n" +
	"      remote                Do not examine local binaries\n" +
	"      force                 Force re-symbolization\n" +
	"    Binary                  Local path or build id of binary for symbolization\n"

var usageMsgVars = "\n\n" +
	"  Misc options:\n" +
	"   -tools                 Search path for object tools\n" +
	"\n" +
	"  Environment Variables:\n" +
	"   PPROF_TMPDIR       Location for saved profiles (default $HOME/pprof)\n" +
	"   PPROF_TOOLS        Search path for object-level tools\n" +
	"   PPROF_BINARY_PATH  Search path for local binary files\n" +
	"                      default: $HOME/pprof/binaries\n" +
	"                      finds binaries by $name and $buildid/$name\n" +
	"   * On Windows, %USERPROFILE% is used instead of $HOME"
