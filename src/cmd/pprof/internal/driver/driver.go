// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package driver implements the core pprof functionality. It can be
// parameterized with a flag implementation, fetch and symbolize
// mechanisms.
package driver

import (
	"bytes"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"cmd/pprof/internal/commands"
	"cmd/pprof/internal/plugin"
	"cmd/pprof/internal/profile"
	"cmd/pprof/internal/report"
	"cmd/pprof/internal/tempfile"
)

// PProf acquires a profile, and symbolizes it using a profile
// manager. Then it generates a report formatted according to the
// options selected through the flags package.
func PProf(flagset plugin.FlagSet, fetch plugin.Fetcher, sym plugin.Symbolizer, obj plugin.ObjTool, ui plugin.UI, overrides commands.Commands) error {
	// Remove any temporary files created during pprof processing.
	defer tempfile.Cleanup()

	f, err := getFlags(flagset, overrides, ui)
	if err != nil {
		return err
	}

	obj.SetConfig(*f.flagTools)

	sources := f.profileSource
	if len(sources) > 1 {
		source := sources[0]
		// If the first argument is a supported object file, treat as executable.
		if file, err := obj.Open(source, 0); err == nil {
			file.Close()
			f.profileExecName = source
			sources = sources[1:]
		} else if *f.flagBuildID == "" && isBuildID(source) {
			f.flagBuildID = &source
			sources = sources[1:]
		}
	}

	// errMu protects concurrent accesses to errset and err. errset is set if an
	// error is encountered by one of the goroutines grabbing a profile.
	errMu, errset := sync.Mutex{}, false

	// Fetch profiles.
	wg := sync.WaitGroup{}
	profs := make([]*profile.Profile, len(sources))
	for i, source := range sources {
		wg.Add(1)
		go func(i int, src string) {
			defer wg.Done()
			p, grabErr := grabProfile(src, f.profileExecName, *f.flagBuildID, fetch, sym, obj, ui, f)
			if grabErr != nil {
				errMu.Lock()
				defer errMu.Unlock()
				errset, err = true, grabErr
				return
			}
			profs[i] = p
		}(i, source)
	}
	wg.Wait()
	if errset {
		return err
	}

	// Merge profiles.
	prof := profs[0]
	for _, p := range profs[1:] {
		if err = prof.Merge(p, 1); err != nil {
			return err
		}
	}

	if *f.flagBase != "" {
		// Fetch base profile and subtract from current profile.
		base, err := grabProfile(*f.flagBase, f.profileExecName, *f.flagBuildID, fetch, sym, obj, ui, f)
		if err != nil {
			return err
		}

		if err = prof.Merge(base, -1); err != nil {
			return err
		}
	}

	if err := processFlags(prof, ui, f); err != nil {
		return err
	}

	prof.RemoveUninteresting()

	if *f.flagInteractive {
		return interactive(prof, obj, ui, f)
	}

	return generate(false, prof, obj, ui, f)
}

// isBuildID determines if the profile may contain a build ID, by
// checking that it is a string of hex digits.
func isBuildID(id string) bool {
	return strings.Trim(id, "0123456789abcdefABCDEF") == ""
}

// adjustURL updates the profile source URL based on heuristics. It
// will append ?seconds=sec for CPU profiles if not already
// specified. Returns the hostname if the profile is remote.
func adjustURL(source string, sec int, ui plugin.UI) (adjusted, host string, duration time.Duration) {
	// If there is a local file with this name, just use it.
	if _, err := os.Stat(source); err == nil {
		return source, "", 0
	}

	url, err := url.Parse(source)

	// Automatically add http:// to URLs of the form hostname:port/path.
	// url.Parse treats "hostname" as the Scheme.
	if err != nil || (url.Host == "" && url.Scheme != "" && url.Scheme != "file") {
		url, err = url.Parse("http://" + source)
		if err != nil {
			return source, url.Host, time.Duration(30) * time.Second
		}
	}
	if scheme := strings.ToLower(url.Scheme); scheme == "" || scheme == "file" {
		url.Scheme = ""
		return url.String(), "", 0
	}

	values := url.Query()
	if urlSeconds := values.Get("seconds"); urlSeconds != "" {
		if us, err := strconv.ParseInt(urlSeconds, 10, 32); err == nil {
			if sec >= 0 {
				ui.PrintErr("Overriding -seconds for URL ", source)
			}
			sec = int(us)
		}
	}

	switch strings.ToLower(url.Path) {
	case "", "/":
		// Apply default /profilez.
		url.Path = "/profilez"
	case "/protoz":
		// Rewrite to /profilez?type=proto
		url.Path = "/profilez"
		values.Set("type", "proto")
	}

	if hasDuration(url.Path) {
		if sec > 0 {
			duration = time.Duration(sec) * time.Second
			values.Set("seconds", fmt.Sprintf("%d", sec))
		} else {
			// Assume default duration: 30 seconds
			duration = 30 * time.Second
		}
	}
	url.RawQuery = values.Encode()
	return url.String(), url.Host, duration
}

func hasDuration(path string) bool {
	for _, trigger := range []string{"profilez", "wallz", "/profile"} {
		if strings.Contains(path, trigger) {
			return true
		}
	}
	return false
}

// preprocess does filtering and aggregation of a profile based on the
// requested options.
func preprocess(prof *profile.Profile, ui plugin.UI, f *flags) error {
	if *f.flagFocus != "" || *f.flagIgnore != "" || *f.flagHide != "" {
		focus, ignore, hide, err := compileFocusIgnore(*f.flagFocus, *f.flagIgnore, *f.flagHide)
		if err != nil {
			return err
		}
		fm, im, hm := prof.FilterSamplesByName(focus, ignore, hide)

		warnNoMatches(fm, *f.flagFocus, "Focus", ui)
		warnNoMatches(im, *f.flagIgnore, "Ignore", ui)
		warnNoMatches(hm, *f.flagHide, "Hide", ui)
	}

	if *f.flagTagFocus != "" || *f.flagTagIgnore != "" {
		focus, err := compileTagFilter(*f.flagTagFocus, ui)
		if err != nil {
			return err
		}
		ignore, err := compileTagFilter(*f.flagTagIgnore, ui)
		if err != nil {
			return err
		}
		fm, im := prof.FilterSamplesByTag(focus, ignore)

		warnNoMatches(fm, *f.flagTagFocus, "TagFocus", ui)
		warnNoMatches(im, *f.flagTagIgnore, "TagIgnore", ui)
	}

	return aggregate(prof, f)
}

func compileFocusIgnore(focus, ignore, hide string) (f, i, h *regexp.Regexp, err error) {
	if focus != "" {
		if f, err = regexp.Compile(focus); err != nil {
			return nil, nil, nil, fmt.Errorf("parsing focus regexp: %v", err)
		}
	}

	if ignore != "" {
		if i, err = regexp.Compile(ignore); err != nil {
			return nil, nil, nil, fmt.Errorf("parsing ignore regexp: %v", err)
		}
	}

	if hide != "" {
		if h, err = regexp.Compile(hide); err != nil {
			return nil, nil, nil, fmt.Errorf("parsing hide regexp: %v", err)
		}
	}
	return
}

func compileTagFilter(filter string, ui plugin.UI) (f func(string, string, int64) bool, err error) {
	if filter == "" {
		return nil, nil
	}
	if numFilter := parseTagFilterRange(filter); numFilter != nil {
		ui.PrintErr("Interpreted '", filter, "' as range, not regexp")
		return func(key, val string, num int64) bool {
			if val != "" {
				return false
			}
			return numFilter(num, key)
		}, nil
	}
	fx, err := regexp.Compile(filter)
	if err != nil {
		return nil, err
	}

	return func(key, val string, num int64) bool {
		if val == "" {
			return false
		}
		return fx.MatchString(key + ":" + val)
	}, nil
}

var tagFilterRangeRx = regexp.MustCompile("([[:digit:]]+)([[:alpha:]]+)")

// parseTagFilterRange returns a function to checks if a value is
// contained on the range described by a string. It can recognize
// strings of the form:
// "32kb" -- matches values == 32kb
// ":64kb" -- matches values <= 64kb
// "4mb:" -- matches values >= 4mb
// "12kb:64mb" -- matches values between 12kb and 64mb (both included).
func parseTagFilterRange(filter string) func(int64, string) bool {
	ranges := tagFilterRangeRx.FindAllStringSubmatch(filter, 2)
	if len(ranges) == 0 {
		return nil // No ranges were identified
	}
	v, err := strconv.ParseInt(ranges[0][1], 10, 64)
	if err != nil {
		panic(fmt.Errorf("Failed to parse int %s: %v", ranges[0][1], err))
	}
	value, unit := report.ScaleValue(v, ranges[0][2], ranges[0][2])
	if len(ranges) == 1 {
		switch match := ranges[0][0]; filter {
		case match:
			return func(v int64, u string) bool {
				sv, su := report.ScaleValue(v, u, unit)
				return su == unit && sv == value
			}
		case match + ":":
			return func(v int64, u string) bool {
				sv, su := report.ScaleValue(v, u, unit)
				return su == unit && sv >= value
			}
		case ":" + match:
			return func(v int64, u string) bool {
				sv, su := report.ScaleValue(v, u, unit)
				return su == unit && sv <= value
			}
		}
		return nil
	}
	if filter != ranges[0][0]+":"+ranges[1][0] {
		return nil
	}
	if v, err = strconv.ParseInt(ranges[1][1], 10, 64); err != nil {
		panic(fmt.Errorf("Failed to parse int %s: %v", ranges[1][1], err))
	}
	value2, unit2 := report.ScaleValue(v, ranges[1][2], unit)
	if unit != unit2 {
		return nil
	}
	return func(v int64, u string) bool {
		sv, su := report.ScaleValue(v, u, unit)
		return su == unit && sv >= value && sv <= value2
	}
}

func warnNoMatches(match bool, rx, option string, ui plugin.UI) {
	if !match && rx != "" && rx != "." {
		ui.PrintErr(option + " expression matched no samples: " + rx)
	}
}

// grabProfile fetches and symbolizes a profile.
func grabProfile(source, exec, buildid string, fetch plugin.Fetcher, sym plugin.Symbolizer, obj plugin.ObjTool, ui plugin.UI, f *flags) (*profile.Profile, error) {
	source, host, duration := adjustURL(source, *f.flagSeconds, ui)
	remote := host != ""

	if remote {
		ui.Print("Fetching profile from ", source)
		if duration != 0 {
			ui.Print("Please wait... (" + duration.String() + ")")
		}
	}

	now := time.Now()
	// Fetch profile from source.
	// Give 50% slack on the timeout.
	p, err := fetch(source, duration+duration/2, ui)
	if err != nil {
		return nil, err
	}

	// Update the time/duration if the profile source doesn't include it.
	// TODO(rsilvera): Remove this when we remove support for legacy profiles.
	if remote {
		if p.TimeNanos == 0 {
			p.TimeNanos = now.UnixNano()
		}
		if duration != 0 && p.DurationNanos == 0 {
			p.DurationNanos = int64(duration)
		}
	}

	// Replace executable/buildID with the options provided in the
	// command line. Assume the executable is the first Mapping entry.
	if exec != "" || buildid != "" {
		if len(p.Mapping) == 0 {
			// Create a fake mapping to hold the user option, and associate
			// all samples to it.
			m := &profile.Mapping{
				ID: 1,
			}
			for _, l := range p.Location {
				l.Mapping = m
			}
			p.Mapping = []*profile.Mapping{m}
		}
		if exec != "" {
			p.Mapping[0].File = exec
		}
		if buildid != "" {
			p.Mapping[0].BuildID = buildid
		}
	}

	if err := sym(*f.flagSymbolize, source, p, obj, ui); err != nil {
		return nil, err
	}

	// Save a copy of any remote profiles, unless the user is explicitly
	// saving it.
	if remote && !f.isFormat("proto") {
		prefix := "pprof."
		if len(p.Mapping) > 0 && p.Mapping[0].File != "" {
			prefix = prefix + filepath.Base(p.Mapping[0].File) + "."
		}
		if !strings.ContainsRune(host, os.PathSeparator) {
			prefix = prefix + host + "."
		}
		for _, s := range p.SampleType {
			prefix = prefix + s.Type + "."
		}

		dir := os.Getenv("PPROF_TMPDIR")
		tempFile, err := tempfile.New(dir, prefix, ".pb.gz")
		if err == nil {
			if err = p.Write(tempFile); err == nil {
				ui.PrintErr("Saved profile in ", tempFile.Name())
			}
		}
		if err != nil {
			ui.PrintErr("Could not save profile: ", err)
		}
	}

	if err := p.Demangle(obj.Demangle); err != nil {
		ui.PrintErr("Failed to demangle profile: ", err)
	}

	if err := p.CheckValid(); err != nil {
		return nil, fmt.Errorf("Grab %s: %v", source, err)
	}

	return p, nil
}

type flags struct {
	flagInteractive   *bool              // Accept commands interactively
	flagCommands      map[string]*bool   // pprof commands without parameters
	flagParamCommands map[string]*string // pprof commands with parameters

	flagSVGPan *string // URL to fetch the SVG Pan library
	flagOutput *string // Output file name

	flagCum      *bool // Sort by cumulative data
	flagCallTree *bool // generate a context-sensitive call tree

	flagAddresses *bool // Report at address level
	flagLines     *bool // Report at source line level
	flagFiles     *bool // Report at file level
	flagFunctions *bool // Report at function level [default]

	flagSymbolize *string // Symbolization options (=none to disable)
	flagBuildID   *string // Override build if for first mapping

	flagNodeCount    *int     // Max number of nodes to show
	flagNodeFraction *float64 // Hide nodes below <f>*total
	flagEdgeFraction *float64 // Hide edges below <f>*total
	flagTrim         *bool    // Set to false to ignore NodeCount/*Fraction
	flagFocus        *string  // Restricts to paths going through a node matching regexp
	flagIgnore       *string  // Skips paths going through any nodes matching regexp
	flagHide         *string  // Skips sample locations matching regexp
	flagTagFocus     *string  // Restrict to samples tagged with key:value matching regexp
	flagTagIgnore    *string  // Discard samples tagged with key:value matching regexp
	flagDropNegative *bool    // Skip negative values

	flagBase *string // Source for base profile to user for comparison

	flagSeconds *int // Length of time for dynamic profiles

	flagTotalDelay  *bool // Display total delay at each region
	flagContentions *bool // Display number of delays at each region
	flagMeanDelay   *bool // Display mean delay at each region

	flagInUseSpace   *bool    // Display in-use memory size
	flagInUseObjects *bool    // Display in-use object counts
	flagAllocSpace   *bool    // Display allocated memory size
	flagAllocObjects *bool    // Display allocated object counts
	flagDisplayUnit  *string  // Measurement unit to use on reports
	flagDivideBy     *float64 // Ratio to divide sample values

	flagSampleIndex *int  // Sample value to use in reports.
	flagMean        *bool // Use mean of sample_index over count

	flagTools       *string
	profileSource   []string
	profileExecName string

	extraUsage string
	commands   commands.Commands
}

func (f *flags) isFormat(format string) bool {
	if fl := f.flagCommands[format]; fl != nil {
		return *fl
	}
	if fl := f.flagParamCommands[format]; fl != nil {
		return *fl != ""
	}
	return false
}

// String provides a printable representation for the current set of flags.
func (f *flags) String(p *profile.Profile) string {
	var ret string

	if ix := *f.flagSampleIndex; ix != -1 {
		ret += fmt.Sprintf("  %-25s : %d (%s)\n", "sample_index", ix, p.SampleType[ix].Type)
	}
	if ix := *f.flagMean; ix {
		ret += boolFlagString("mean")
	}
	if *f.flagDisplayUnit != "minimum" {
		ret += stringFlagString("unit", *f.flagDisplayUnit)
	}

	switch {
	case *f.flagInteractive:
		ret += boolFlagString("interactive")
	}
	for name, fl := range f.flagCommands {
		if *fl {
			ret += boolFlagString(name)
		}
	}

	if *f.flagCum {
		ret += boolFlagString("cum")
	}
	if *f.flagCallTree {
		ret += boolFlagString("call_tree")
	}

	switch {
	case *f.flagAddresses:
		ret += boolFlagString("addresses")
	case *f.flagLines:
		ret += boolFlagString("lines")
	case *f.flagFiles:
		ret += boolFlagString("files")
	case *f.flagFunctions:
		ret += boolFlagString("functions")
	}

	if *f.flagNodeCount != -1 {
		ret += intFlagString("nodecount", *f.flagNodeCount)
	}

	ret += floatFlagString("nodefraction", *f.flagNodeFraction)
	ret += floatFlagString("edgefraction", *f.flagEdgeFraction)

	if *f.flagFocus != "" {
		ret += stringFlagString("focus", *f.flagFocus)
	}
	if *f.flagIgnore != "" {
		ret += stringFlagString("ignore", *f.flagIgnore)
	}
	if *f.flagHide != "" {
		ret += stringFlagString("hide", *f.flagHide)
	}

	if *f.flagTagFocus != "" {
		ret += stringFlagString("tagfocus", *f.flagTagFocus)
	}
	if *f.flagTagIgnore != "" {
		ret += stringFlagString("tagignore", *f.flagTagIgnore)
	}

	return ret
}

func boolFlagString(label string) string {
	return fmt.Sprintf("  %-25s : true\n", label)
}

func stringFlagString(label, value string) string {
	return fmt.Sprintf("  %-25s : %s\n", label, value)
}

func intFlagString(label string, value int) string {
	return fmt.Sprintf("  %-25s : %d\n", label, value)
}

func floatFlagString(label string, value float64) string {
	return fmt.Sprintf("  %-25s : %f\n", label, value)
}

// Utility routines to set flag values.
func newBool(b bool) *bool {
	return &b
}

func newString(s string) *string {
	return &s
}

func newFloat64(fl float64) *float64 {
	return &fl
}

func newInt(i int) *int {
	return &i
}

func (f *flags) usage(ui plugin.UI) {
	var commandMsg []string
	for name, cmd := range f.commands {
		if cmd.HasParam {
			name = name + "=p"
		}
		commandMsg = append(commandMsg,
			fmt.Sprintf("  -%-16s %s", name, cmd.Usage))
	}

	sort.Strings(commandMsg)

	text := usageMsgHdr + strings.Join(commandMsg, "\n") + "\n" + usageMsg + "\n"
	if f.extraUsage != "" {
		text += f.extraUsage + "\n"
	}
	text += usageMsgVars
	ui.Print(text)
}

func getFlags(flag plugin.FlagSet, overrides commands.Commands, ui plugin.UI) (*flags, error) {
	f := &flags{
		flagInteractive:   flag.Bool("interactive", false, "Accepts commands interactively"),
		flagCommands:      make(map[string]*bool),
		flagParamCommands: make(map[string]*string),

		// Filename for file-based output formats, stdout by default.
		flagOutput: flag.String("output", "", "Output filename for file-based outputs "),
		// Comparisons.
		flagBase:         flag.String("base", "", "Source for base profile for comparison"),
		flagDropNegative: flag.Bool("drop_negative", false, "Ignore negative differences"),

		flagSVGPan: flag.String("svgpan", "https://www.cyberz.org/projects/SVGPan/SVGPan.js", "URL for SVGPan Library"),
		// Data sorting criteria.
		flagCum: flag.Bool("cum", false, "Sort by cumulative data"),
		// Graph handling options.
		flagCallTree: flag.Bool("call_tree", false, "Create a context-sensitive call tree"),
		// Granularity of output resolution.
		flagAddresses: flag.Bool("addresses", false, "Report at address level"),
		flagLines:     flag.Bool("lines", false, "Report at source line level"),
		flagFiles:     flag.Bool("files", false, "Report at source file level"),
		flagFunctions: flag.Bool("functions", false, "Report at function level [default]"),
		// Internal options.
		flagSymbolize: flag.String("symbolize", "", "Options for profile symbolization"),
		flagBuildID:   flag.String("buildid", "", "Override build id for first mapping"),
		// Filtering options
		flagNodeCount:    flag.Int("nodecount", -1, "Max number of nodes to show"),
		flagNodeFraction: flag.Float64("nodefraction", 0.005, "Hide nodes below <f>*total"),
		flagEdgeFraction: flag.Float64("edgefraction", 0.001, "Hide edges below <f>*total"),
		flagTrim:         flag.Bool("trim", true, "Honor nodefraction/edgefraction/nodecount defaults"),
		flagFocus:        flag.String("focus", "", "Restricts to paths going through a node matching regexp"),
		flagIgnore:       flag.String("ignore", "", "Skips paths going through any nodes matching regexp"),
		flagHide:         flag.String("hide", "", "Skips nodes matching regexp"),
		flagTagFocus:     flag.String("tagfocus", "", "Restrict to samples with tags in range or matched by regexp"),
		flagTagIgnore:    flag.String("tagignore", "", "Discard samples with tags in range or matched by regexp"),
		// CPU profile options
		flagSeconds: flag.Int("seconds", -1, "Length of time for dynamic profiles"),
		// Heap profile options
		flagInUseSpace:   flag.Bool("inuse_space", false, "Display in-use memory size"),
		flagInUseObjects: flag.Bool("inuse_objects", false, "Display in-use object counts"),
		flagAllocSpace:   flag.Bool("alloc_space", false, "Display allocated memory size"),
		flagAllocObjects: flag.Bool("alloc_objects", false, "Display allocated object counts"),
		flagDisplayUnit:  flag.String("unit", "minimum", "Measurement units to display"),
		flagDivideBy:     flag.Float64("divide_by", 1.0, "Ratio to divide all samples before visualization"),
		flagSampleIndex:  flag.Int("sample_index", -1, "Index of sample value to report"),
		flagMean:         flag.Bool("mean", false, "Average sample value over first value (count)"),
		// Contention profile options
		flagTotalDelay:  flag.Bool("total_delay", false, "Display total delay at each region"),
		flagContentions: flag.Bool("contentions", false, "Display number of delays at each region"),
		flagMeanDelay:   flag.Bool("mean_delay", false, "Display mean delay at each region"),
		flagTools:       flag.String("tools", os.Getenv("PPROF_TOOLS"), "Path for object tool pathnames"),
		extraUsage:      flag.ExtraUsage(),
	}

	// Flags used during command processing
	interactive := &f.flagInteractive
	svgpan := &f.flagSVGPan
	f.commands = commands.PProf(functionCompleter, interactive, svgpan)

	// Override commands
	for name, cmd := range overrides {
		f.commands[name] = cmd
	}

	for name, cmd := range f.commands {
		if cmd.HasParam {
			f.flagParamCommands[name] = flag.String(name, "", "Generate a report in "+name+" format, matching regexp")
		} else {
			f.flagCommands[name] = flag.Bool(name, false, "Generate a report in "+name+" format")
		}
	}

	args := flag.Parse(func() { f.usage(ui) })
	if len(args) == 0 {
		return nil, fmt.Errorf("no profile source specified")
	}

	f.profileSource = args

	// Instruct legacy heapz parsers to grab historical allocation data,
	// instead of the default in-use data. Not available with tcmalloc.
	if *f.flagAllocSpace || *f.flagAllocObjects {
		profile.LegacyHeapAllocated = true
	}

	if profileDir := os.Getenv("PPROF_TMPDIR"); profileDir == "" {
		profileDir = os.Getenv("HOME") + "/pprof"
		os.Setenv("PPROF_TMPDIR", profileDir)
		if err := os.MkdirAll(profileDir, 0755); err != nil {
			return nil, fmt.Errorf("failed to access temp dir %s: %v", profileDir, err)
		}
	}

	return f, nil
}

func processFlags(p *profile.Profile, ui plugin.UI, f *flags) error {
	flagDis := f.isFormat("disasm")
	flagPeek := f.isFormat("peek")
	flagWebList := f.isFormat("weblist")
	flagList := f.isFormat("list")

	if flagDis || flagWebList {
		// Collect all samples at address granularity for assembly
		// listing.
		f.flagNodeCount = newInt(0)
		f.flagAddresses = newBool(true)
		f.flagLines = newBool(false)
		f.flagFiles = newBool(false)
		f.flagFunctions = newBool(false)
	}

	if flagPeek {
		// Collect all samples at function granularity for peek command
		f.flagNodeCount = newInt(0)
		f.flagAddresses = newBool(false)
		f.flagLines = newBool(false)
		f.flagFiles = newBool(false)
		f.flagFunctions = newBool(true)
	}

	if flagList {
		// Collect all samples at fileline granularity for source
		// listing.
		f.flagNodeCount = newInt(0)
		f.flagAddresses = newBool(false)
		f.flagLines = newBool(true)
		f.flagFiles = newBool(false)
		f.flagFunctions = newBool(false)
	}

	if !*f.flagTrim {
		f.flagNodeCount = newInt(0)
		f.flagNodeFraction = newFloat64(0)
		f.flagEdgeFraction = newFloat64(0)
	}

	if oc := countFlagMap(f.flagCommands, f.flagParamCommands); oc == 0 {
		f.flagInteractive = newBool(true)
	} else if oc > 1 {
		f.usage(ui)
		return fmt.Errorf("must set at most one output format")
	}

	// Apply nodecount defaults for non-interactive mode. The
	// interactive shell will apply defaults for the interactive mode.
	if *f.flagNodeCount < 0 && !*f.flagInteractive {
		switch {
		default:
			f.flagNodeCount = newInt(80)
		case f.isFormat("text"):
			f.flagNodeCount = newInt(0)
		}
	}

	// Apply legacy options and diagnose conflicts.
	if rc := countFlags([]*bool{f.flagAddresses, f.flagLines, f.flagFiles, f.flagFunctions}); rc == 0 {
		f.flagFunctions = newBool(true)
	} else if rc > 1 {
		f.usage(ui)
		return fmt.Errorf("must set at most one granularity option")
	}

	var err error
	si, sm := *f.flagSampleIndex, *f.flagMean || *f.flagMeanDelay
	si, err = sampleIndex(p, &f.flagTotalDelay, si, 1, "delay", "-total_delay", err)
	si, err = sampleIndex(p, &f.flagMeanDelay, si, 1, "delay", "-mean_delay", err)
	si, err = sampleIndex(p, &f.flagContentions, si, 0, "contentions", "-contentions", err)

	si, err = sampleIndex(p, &f.flagInUseSpace, si, 1, "inuse_space", "-inuse_space", err)
	si, err = sampleIndex(p, &f.flagInUseObjects, si, 0, "inuse_objects", "-inuse_objects", err)
	si, err = sampleIndex(p, &f.flagAllocSpace, si, 1, "alloc_space", "-alloc_space", err)
	si, err = sampleIndex(p, &f.flagAllocObjects, si, 0, "alloc_objects", "-alloc_objects", err)

	if si == -1 {
		// Use last value if none is requested.
		si = len(p.SampleType) - 1
	} else if si < 0 || si >= len(p.SampleType) {
		err = fmt.Errorf("sample_index value %d out of range [0..%d]", si, len(p.SampleType)-1)
	}

	if err != nil {
		f.usage(ui)
		return err
	}
	f.flagSampleIndex, f.flagMean = newInt(si), newBool(sm)
	return nil
}

func sampleIndex(p *profile.Profile, flag **bool,
	sampleIndex int,
	newSampleIndex int,
	sampleType, option string,
	err error) (int, error) {
	if err != nil || !**flag {
		return sampleIndex, err
	}
	*flag = newBool(false)
	if sampleIndex != -1 {
		return 0, fmt.Errorf("set at most one sample value selection option")
	}
	if newSampleIndex >= len(p.SampleType) ||
		p.SampleType[newSampleIndex].Type != sampleType {
		return 0, fmt.Errorf("option %s not valid for this profile", option)
	}
	return newSampleIndex, nil
}

func countFlags(bs []*bool) int {
	var c int
	for _, b := range bs {
		if *b {
			c++
		}
	}
	return c
}

func countFlagMap(bms map[string]*bool, bmrxs map[string]*string) int {
	var c int
	for _, b := range bms {
		if *b {
			c++
		}
	}
	for _, s := range bmrxs {
		if *s != "" {
			c++
		}
	}
	return c
}

var usageMsgHdr = "usage: pprof [options] [binary] <profile source> ...\n" +
	"Output format (only set one):\n"

var usageMsg = "Output file parameters (for file-based output formats):\n" +
	"  -output=f         Generate output on file f (stdout by default)\n" +
	"Output granularity (only set one):\n" +
	"  -functions        Report at function level [default]\n" +
	"  -files            Report at source file level\n" +
	"  -lines            Report at source line level\n" +
	"  -addresses        Report at address level\n" +
	"Comparison options:\n" +
	"  -base <profile>   Show delta from this profile\n" +
	"  -drop_negative    Ignore negative differences\n" +
	"Sorting options:\n" +
	"  -cum              Sort by cumulative data\n\n" +
	"Dynamic profile options:\n" +
	"  -seconds=N        Length of time for dynamic profiles\n" +
	"Profile trimming options:\n" +
	"  -nodecount=N      Max number of nodes to show\n" +
	"  -nodefraction=f   Hide nodes below <f>*total\n" +
	"  -edgefraction=f   Hide edges below <f>*total\n" +
	"Sample value selection option (by index):\n" +
	"  -sample_index      Index of sample value to display\n" +
	"  -mean              Average sample value over first value\n" +
	"Sample value selection option (for heap profiles):\n" +
	"  -inuse_space      Display in-use memory size\n" +
	"  -inuse_objects    Display in-use object counts\n" +
	"  -alloc_space      Display allocated memory size\n" +
	"  -alloc_objects    Display allocated object counts\n" +
	"Sample value selection option (for contention profiles):\n" +
	"  -total_delay      Display total delay at each region\n" +
	"  -contentions      Display number of delays at each region\n" +
	"  -mean_delay       Display mean delay at each region\n" +
	"Filtering options:\n" +
	"  -focus=r          Restricts to paths going through a node matching regexp\n" +
	"  -ignore=r         Skips paths going through any nodes matching regexp\n" +
	"  -tagfocus=r       Restrict to samples tagged with key:value matching regexp\n" +
	"                    Restrict to samples with numeric tags in range (eg \"32kb:1mb\")\n" +
	"  -tagignore=r      Discard samples tagged with key:value matching regexp\n" +
	"                    Avoid samples with numeric tags in range (eg \"1mb:\")\n" +
	"Miscellaneous:\n" +
	"  -call_tree        Generate a context-sensitive call tree\n" +
	"  -unit=u           Convert all samples to unit u for display\n" +
	"  -divide_by=f      Scale all samples by dividing them by f\n" +
	"  -buildid=id       Override build id for main binary in profile\n" +
	"  -tools=path       Search path for object-level tools\n" +
	"  -help             This message"

var usageMsgVars = "Environment Variables:\n" +
	"   PPROF_TMPDIR       Location for temporary files (default $HOME/pprof)\n" +
	"   PPROF_TOOLS        Search path for object-level tools\n" +
	"   PPROF_BINARY_PATH  Search path for local binary files\n" +
	"                      default: $HOME/pprof/binaries\n" +
	"                      finds binaries by $name and $buildid/$name"

func aggregate(prof *profile.Profile, f *flags) error {
	switch {
	case f.isFormat("proto"), f.isFormat("raw"):
		// No aggregation for raw profiles.
	case f.isFormat("callgrind"):
		// Aggregate to file/line for callgrind.
		fallthrough
	case *f.flagLines:
		return prof.Aggregate(true, true, true, true, false)
	case *f.flagFiles:
		return prof.Aggregate(true, false, true, false, false)
	case *f.flagFunctions:
		return prof.Aggregate(true, true, false, false, false)
	case f.isFormat("weblist"), f.isFormat("disasm"):
		return prof.Aggregate(false, true, true, true, true)
	}
	return nil
}

// parseOptions parses the options into report.Options
// Returns a function to postprocess the report after generation.
func parseOptions(f *flags) (o *report.Options, p commands.PostProcessor, err error) {

	if *f.flagDivideBy == 0 {
		return nil, nil, fmt.Errorf("zero divisor specified")
	}

	o = &report.Options{
		CumSort:        *f.flagCum,
		CallTree:       *f.flagCallTree,
		PrintAddresses: *f.flagAddresses,
		DropNegative:   *f.flagDropNegative,
		Ratio:          1 / *f.flagDivideBy,

		NodeCount:    *f.flagNodeCount,
		NodeFraction: *f.flagNodeFraction,
		EdgeFraction: *f.flagEdgeFraction,
		OutputUnit:   *f.flagDisplayUnit,
	}

	for cmd, b := range f.flagCommands {
		if *b {
			pcmd := f.commands[cmd]
			o.OutputFormat = pcmd.Format
			return o, pcmd.PostProcess, nil
		}
	}

	for cmd, rx := range f.flagParamCommands {
		if *rx != "" {
			pcmd := f.commands[cmd]
			if o.Symbol, err = regexp.Compile(*rx); err != nil {
				return nil, nil, fmt.Errorf("parsing -%s regexp: %v", cmd, err)
			}
			o.OutputFormat = pcmd.Format
			return o, pcmd.PostProcess, nil
		}
	}

	return nil, nil, fmt.Errorf("no output format selected")
}

type sampleValueFunc func(*profile.Sample) int64

// sampleFormat returns a function to extract values out of a profile.Sample,
// and the type/units of those values.
func sampleFormat(p *profile.Profile, f *flags) (sampleValueFunc, string, string) {
	valueIndex := *f.flagSampleIndex

	if *f.flagMean {
		return meanExtractor(valueIndex), "mean_" + p.SampleType[valueIndex].Type, p.SampleType[valueIndex].Unit
	}

	return valueExtractor(valueIndex), p.SampleType[valueIndex].Type, p.SampleType[valueIndex].Unit
}

func valueExtractor(ix int) sampleValueFunc {
	return func(s *profile.Sample) int64 {
		return s.Value[ix]
	}
}

func meanExtractor(ix int) sampleValueFunc {
	return func(s *profile.Sample) int64 {
		if s.Value[0] == 0 {
			return 0
		}
		return s.Value[ix] / s.Value[0]
	}
}

func generate(interactive bool, prof *profile.Profile, obj plugin.ObjTool, ui plugin.UI, f *flags) error {
	o, postProcess, err := parseOptions(f)
	if err != nil {
		return err
	}

	var w io.Writer
	if *f.flagOutput == "" {
		w = os.Stdout
	} else {
		ui.PrintErr("Generating report in ", *f.flagOutput)
		outputFile, err := os.Create(*f.flagOutput)
		if err != nil {
			return err
		}
		defer outputFile.Close()
		w = outputFile
	}

	value, stype, unit := sampleFormat(prof, f)
	o.SampleType = stype
	rpt := report.New(prof, *o, value, unit)

	// Do not apply filters if we're just generating a proto, so we
	// still have all the data.
	if o.OutputFormat != report.Proto {
		// Delay applying focus/ignore until after creating the report so
		// the report reflects the total number of samples.
		if err := preprocess(prof, ui, f); err != nil {
			return err
		}
	}

	if postProcess == nil {
		return report.Generate(w, rpt, obj)
	}

	var dot bytes.Buffer
	if err = report.Generate(&dot, rpt, obj); err != nil {
		return err
	}

	return postProcess(&dot, w, ui)
}
