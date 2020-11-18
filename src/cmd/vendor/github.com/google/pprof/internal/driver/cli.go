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
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/google/pprof/internal/binutils"
	"github.com/google/pprof/internal/plugin"
)

type source struct {
	Sources   []string
	ExecName  string
	BuildID   string
	Base      []string
	DiffBase  bool
	Normalize bool

	Seconds            int
	Timeout            int
	Symbolize          string
	HTTPHostport       string
	HTTPDisableBrowser bool
	Comment            string
}

// parseFlags parses the command lines through the specified flags package
// and returns the source of the profile and optionally the command
// for the kind of report to generate (nil for interactive use).
func parseFlags(o *plugin.Options) (*source, []string, error) {
	flag := o.Flagset
	// Comparisons.
	flagDiffBase := flag.StringList("diff_base", "", "Source of base profile for comparison")
	flagBase := flag.StringList("base", "", "Source of base profile for profile subtraction")
	// Source options.
	flagSymbolize := flag.String("symbolize", "", "Options for profile symbolization")
	flagBuildID := flag.String("buildid", "", "Override build id for first mapping")
	flagTimeout := flag.Int("timeout", -1, "Timeout in seconds for fetching a profile")
	flagAddComment := flag.String("add_comment", "", "Annotation string to record in the profile")
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

	flagHTTP := flag.String("http", "", "Present interactive web UI at the specified http host:port")
	flagNoBrowser := flag.Bool("no_browser", false, "Skip opening a browswer for the interactive web UI")

	// Flags that set configuration properties.
	cfg := currentConfig()
	configFlagSetter := installConfigFlags(flag, &cfg)

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
		return nil, nil, errors.New("no profile source specified")
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

	// Apply any specified flags to cfg.
	if err := configFlagSetter(); err != nil {
		return nil, nil, err
	}

	cmd, err := outputFormat(flagCommands, flagParamCommands)
	if err != nil {
		return nil, nil, err
	}
	if cmd != nil && *flagHTTP != "" {
		return nil, nil, errors.New("-http is not compatible with an output format on the command line")
	}

	if *flagNoBrowser && *flagHTTP == "" {
		return nil, nil, errors.New("-no_browser only makes sense with -http")
	}

	si := cfg.SampleIndex
	si = sampleIndex(flagTotalDelay, si, "delay", "-total_delay", o.UI)
	si = sampleIndex(flagMeanDelay, si, "delay", "-mean_delay", o.UI)
	si = sampleIndex(flagContentions, si, "contentions", "-contentions", o.UI)
	si = sampleIndex(flagInUseSpace, si, "inuse_space", "-inuse_space", o.UI)
	si = sampleIndex(flagInUseObjects, si, "inuse_objects", "-inuse_objects", o.UI)
	si = sampleIndex(flagAllocSpace, si, "alloc_space", "-alloc_space", o.UI)
	si = sampleIndex(flagAllocObjects, si, "alloc_objects", "-alloc_objects", o.UI)
	cfg.SampleIndex = si

	if *flagMeanDelay {
		cfg.Mean = true
	}

	source := &source{
		Sources:            args,
		ExecName:           execName,
		BuildID:            *flagBuildID,
		Seconds:            *flagSeconds,
		Timeout:            *flagTimeout,
		Symbolize:          *flagSymbolize,
		HTTPHostport:       *flagHTTP,
		HTTPDisableBrowser: *flagNoBrowser,
		Comment:            *flagAddComment,
	}

	if err := source.addBaseProfiles(*flagBase, *flagDiffBase); err != nil {
		return nil, nil, err
	}

	normalize := cfg.Normalize
	if normalize && len(source.Base) == 0 {
		return nil, nil, errors.New("must have base profile to normalize by")
	}
	source.Normalize = normalize

	if bu, ok := o.Obj.(*binutils.Binutils); ok {
		bu.SetTools(*flagTools)
	}

	setCurrentConfig(cfg)
	return source, cmd, nil
}

// addBaseProfiles adds the list of base profiles or diff base profiles to
// the source. This function will return an error if both base and diff base
// profiles are specified.
func (source *source) addBaseProfiles(flagBase, flagDiffBase []*string) error {
	base, diffBase := dropEmpty(flagBase), dropEmpty(flagDiffBase)
	if len(base) > 0 && len(diffBase) > 0 {
		return errors.New("-base and -diff_base flags cannot both be specified")
	}

	source.Base = base
	if len(diffBase) > 0 {
		source.Base, source.DiffBase = diffBase, true
	}
	return nil
}

// dropEmpty list takes a slice of string pointers, and outputs a slice of
// non-empty strings associated with the flag.
func dropEmpty(list []*string) []string {
	var l []string
	for _, s := range list {
		if *s != "" {
			l = append(l, *s)
		}
	}
	return l
}

// installConfigFlags creates command line flags for configuration
// fields and returns a function which can be called after flags have
// been parsed to copy any flags specified on the command line to
// *cfg.
func installConfigFlags(flag plugin.FlagSet, cfg *config) func() error {
	// List of functions for setting the different parts of a config.
	var setters []func()
	var err error // Holds any errors encountered while running setters.

	for _, field := range configFields {
		n := field.name
		help := configHelp[n]
		var setter func()
		switch ptr := cfg.fieldPtr(field).(type) {
		case *bool:
			f := flag.Bool(n, *ptr, help)
			setter = func() { *ptr = *f }
		case *int:
			f := flag.Int(n, *ptr, help)
			setter = func() { *ptr = *f }
		case *float64:
			f := flag.Float64(n, *ptr, help)
			setter = func() { *ptr = *f }
		case *string:
			if len(field.choices) == 0 {
				f := flag.String(n, *ptr, help)
				setter = func() { *ptr = *f }
			} else {
				// Make a separate flag per possible choice.
				// Set all flags to initially false so we can
				// identify conflicts.
				bools := make(map[string]*bool)
				for _, choice := range field.choices {
					bools[choice] = flag.Bool(choice, false, configHelp[choice])
				}
				setter = func() {
					var set []string
					for k, v := range bools {
						if *v {
							set = append(set, k)
						}
					}
					switch len(set) {
					case 0:
						// Leave as default value.
					case 1:
						*ptr = set[0]
					default:
						err = fmt.Errorf("conflicting options set: %v", set)
					}
				}
			}
		}
		setters = append(setters, setter)
	}

	return func() error {
		// Apply the setter for every flag.
		for _, setter := range setters {
			setter()
			if err != nil {
				return err
			}
		}
		return nil
	}
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
				return nil, errors.New("must set at most one output format")
			}
			cmd = []string{n}
		}
	}
	for n, s := range acmd {
		if *s != "" {
			if cmd != nil {
				return nil, errors.New("must set at most one output format")
			}
			cmd = []string{n, *s}
		}
	}
	return cmd, nil
}

var usageMsgHdr = `usage:

Produce output in the specified format.

   pprof <format> [options] [binary] <source> ...

Omit the format to get an interactive shell whose commands can be used
to generate various views of a profile

   pprof [options] [binary] <source> ...

Omit the format and provide the "-http" flag to get an interactive web
interface at the specified host:port that can be used to navigate through
various views of a profile.

   pprof -http [host]:[port] [options] [binary] <source> ...

Details:
`

var usageMsgSrc = "\n\n" +
	"  Source options:\n" +
	"    -seconds              Duration for time-based profile collection\n" +
	"    -timeout              Timeout in seconds for profile collection\n" +
	"    -buildid              Override build id for main binary\n" +
	"    -add_comment          Free-form annotation to add to the profile\n" +
	"                          Displayed on some reports or with pprof -comments\n" +
	"    -diff_base source     Source of base profile for comparison\n" +
	"    -base source          Source of base profile for profile subtraction\n" +
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
	"   -http              Provide web interface at host:port.\n" +
	"                      Host is optional and 'localhost' by default.\n" +
	"                      Port is optional and a randomly available port by default.\n" +
	"   -no_browser        Skip opening a browser for the interactive web UI.\n" +
	"   -tools             Search path for object tools\n" +
	"\n" +
	"  Legacy convenience options:\n" +
	"   -inuse_space           Same as -sample_index=inuse_space\n" +
	"   -inuse_objects         Same as -sample_index=inuse_objects\n" +
	"   -alloc_space           Same as -sample_index=alloc_space\n" +
	"   -alloc_objects         Same as -sample_index=alloc_objects\n" +
	"   -total_delay           Same as -sample_index=delay\n" +
	"   -contentions           Same as -sample_index=contentions\n" +
	"   -mean_delay            Same as -mean -sample_index=delay\n" +
	"\n" +
	"  Environment Variables:\n" +
	"   PPROF_TMPDIR       Location for saved profiles (default $HOME/pprof)\n" +
	"   PPROF_TOOLS        Search path for object-level tools\n" +
	"   PPROF_BINARY_PATH  Search path for local binary files\n" +
	"                      default: $HOME/pprof/binaries\n" +
	"                      searches $name, $path, $buildid/$name, $path/$buildid\n" +
	"   * On Windows, %USERPROFILE% is used instead of $HOME"
