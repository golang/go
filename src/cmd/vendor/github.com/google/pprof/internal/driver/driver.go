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

// Package driver implements the core pprof functionality. It can be
// parameterized with a flag implementation, fetch and symbolize
// mechanisms.
package driver

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/google/pprof/internal/plugin"
	"github.com/google/pprof/internal/report"
	"github.com/google/pprof/profile"
)

// PProf acquires a profile, and symbolizes it using a profile
// manager. Then it generates a report formatted according to the
// options selected through the flags package.
func PProf(eo *plugin.Options) error {
	// Remove any temporary files created during pprof processing.
	defer cleanupTempFiles()

	o := setDefaults(eo)

	src, cmd, err := parseFlags(o)
	if err != nil {
		return err
	}

	p, err := fetchProfiles(src, o)
	if err != nil {
		return err
	}

	if cmd != nil {
		return generateReport(p, cmd, currentConfig(), o)
	}

	if src.HTTPHostport != "" {
		return serveWebInterface(src.HTTPHostport, p, o, src.HTTPDisableBrowser)
	}
	return interactive(p, o)
}

// generateRawReport is allowed to modify p.
func generateRawReport(p *profile.Profile, cmd []string, cfg config, o *plugin.Options) (*command, *report.Report, error) {
	// Identify units of numeric tags in profile.
	numLabelUnits := identifyNumLabelUnits(p, o.UI)

	// Get report output format
	c := pprofCommands[cmd[0]]
	if c == nil {
		panic("unexpected nil command")
	}

	cfg = applyCommandOverrides(cmd[0], c.format, cfg)

	// Create label pseudo nodes before filtering, in case the filters use
	// the generated nodes.
	generateTagRootsLeaves(p, cfg, o.UI)

	// Delay focus after configuring report to get percentages on all samples.
	relative := cfg.RelativePercentages
	if relative {
		if err := applyFocus(p, numLabelUnits, cfg, o.UI); err != nil {
			return nil, nil, err
		}
	}
	ropt, err := reportOptions(p, numLabelUnits, cfg)
	if err != nil {
		return nil, nil, err
	}
	ropt.OutputFormat = c.format
	if len(cmd) == 2 {
		s, err := regexp.Compile(cmd[1])
		if err != nil {
			return nil, nil, fmt.Errorf("parsing argument regexp %s: %v", cmd[1], err)
		}
		ropt.Symbol = s
	}

	rpt := report.New(p, ropt)
	if !relative {
		if err := applyFocus(p, numLabelUnits, cfg, o.UI); err != nil {
			return nil, nil, err
		}
	}
	if err := aggregate(p, cfg); err != nil {
		return nil, nil, err
	}

	return c, rpt, nil
}

// generateReport is allowed to modify p.
func generateReport(p *profile.Profile, cmd []string, cfg config, o *plugin.Options) error {
	c, rpt, err := generateRawReport(p, cmd, cfg, o)
	if err != nil {
		return err
	}

	// Generate the report.
	dst := new(bytes.Buffer)
	switch rpt.OutputFormat() {
	case report.WebList:
		// We need template expansion, so generate here instead of in report.
		err = printWebList(dst, rpt, o.Obj)
	default:
		err = report.Generate(dst, rpt, o.Obj)
	}
	if err != nil {
		return err
	}
	src := dst

	// If necessary, perform any data post-processing.
	if c.postProcess != nil {
		dst = new(bytes.Buffer)
		if err := c.postProcess(src, dst, o.UI); err != nil {
			return err
		}
		src = dst
	}

	// If no output is specified, use default visualizer.
	output := cfg.Output
	if output == "" {
		if c.visualizer != nil {
			return c.visualizer(src, os.Stdout, o.UI)
		}
		_, err := src.WriteTo(os.Stdout)
		return err
	}

	// Output to specified file.
	o.UI.PrintErr("Generating report in ", output)
	out, err := o.Writer.Open(output)
	if err != nil {
		return err
	}
	if _, err := src.WriteTo(out); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

func printWebList(dst io.Writer, rpt *report.Report, obj plugin.ObjTool) error {
	listing, err := report.MakeWebList(rpt, obj, -1)
	if err != nil {
		return err
	}
	legend := report.ProfileLabels(rpt)
	return renderHTML(dst, "sourcelisting", rpt, nil, legend, webArgs{
		Standalone: true,
		Listing:    listing,
	})
}

func applyCommandOverrides(cmd string, outputFormat int, cfg config) config {
	// Some report types override the trim flag to false below. This is to make
	// sure the default heuristics of excluding insignificant nodes and edges
	// from the call graph do not apply. One example where it is important is
	// annotated source or disassembly listing. Those reports run on a specific
	// function (or functions), but the trimming is applied before the function
	// data is selected. So, with trimming enabled, the report could end up
	// showing no data if the specified function is "uninteresting" as far as the
	// trimming is concerned.
	trim := cfg.Trim

	switch cmd {
	case "disasm":
		trim = false
		cfg.Granularity = "addresses"
		// Force the 'noinlines' mode so that source locations for a given address
		// collapse and there is only one for the given address. Without this
		// cumulative metrics would be double-counted when annotating the assembly.
		// This is because the merge is done by address and in case of an inlined
		// stack each of the inlined entries is a separate callgraph node.
		cfg.NoInlines = true
	case "weblist":
		trim = false
		cfg.Granularity = "addresses"
		cfg.NoInlines = false // Need inline info to support call expansion
	case "peek":
		trim = false
	case "list":
		trim = false
		cfg.Granularity = "lines"
		// Do not force 'noinlines' to be false so that specifying
		// "-list foo -noinlines" is supported and works as expected.
	case "text", "top", "topproto":
		if cfg.NodeCount == -1 {
			cfg.NodeCount = 0
		}
	default:
		if cfg.NodeCount == -1 {
			cfg.NodeCount = 80
		}
	}

	switch outputFormat {
	case report.Proto, report.Raw, report.Callgrind:
		trim = false
		cfg.Granularity = "addresses"
	}

	if !trim {
		cfg.NodeCount = 0
		cfg.NodeFraction = 0
		cfg.EdgeFraction = 0
	}
	return cfg
}

// generateTagRootsLeaves generates extra nodes from the tagroot and tagleaf options.
func generateTagRootsLeaves(prof *profile.Profile, cfg config, ui plugin.UI) {
	tagRootLabelKeys := dropEmptyStrings(strings.Split(cfg.TagRoot, ","))
	tagLeafLabelKeys := dropEmptyStrings(strings.Split(cfg.TagLeaf, ","))
	rootm, leafm := addLabelNodes(prof, tagRootLabelKeys, tagLeafLabelKeys, cfg.Unit)
	warnNoMatches(cfg.TagRoot == "" || rootm, "TagRoot", ui)
	warnNoMatches(cfg.TagLeaf == "" || leafm, "TagLeaf", ui)
}

// dropEmptyStrings filters a slice to only non-empty strings
func dropEmptyStrings(in []string) (out []string) {
	for _, s := range in {
		if s != "" {
			out = append(out, s)
		}
	}
	return
}

func aggregate(prof *profile.Profile, cfg config) error {
	var function, filename, linenumber, address bool
	inlines := !cfg.NoInlines
	switch cfg.Granularity {
	case "addresses":
		if inlines {
			return nil
		}
		function = true
		filename = true
		linenumber = true
		address = true
	case "lines":
		function = true
		filename = true
		linenumber = true
	case "files":
		filename = true
	case "functions":
		function = true
	case "filefunctions":
		function = true
		filename = true
	default:
		return fmt.Errorf("unexpected granularity")
	}
	return prof.Aggregate(inlines, function, filename, linenumber, cfg.ShowColumns, address)
}

func reportOptions(p *profile.Profile, numLabelUnits map[string]string, cfg config) (*report.Options, error) {
	si, mean := cfg.SampleIndex, cfg.Mean
	value, meanDiv, sample, err := sampleFormat(p, si, mean)
	if err != nil {
		return nil, err
	}

	stype := sample.Type
	if mean {
		stype = "mean_" + stype
	}

	if cfg.DivideBy == 0 {
		return nil, fmt.Errorf("zero divisor specified")
	}

	var filters []string
	addFilter := func(k string, v string) {
		if v != "" {
			filters = append(filters, k+"="+v)
		}
	}
	addFilter("focus", cfg.Focus)
	addFilter("ignore", cfg.Ignore)
	addFilter("hide", cfg.Hide)
	addFilter("show", cfg.Show)
	addFilter("show_from", cfg.ShowFrom)
	addFilter("tagfocus", cfg.TagFocus)
	addFilter("tagignore", cfg.TagIgnore)
	addFilter("tagshow", cfg.TagShow)
	addFilter("taghide", cfg.TagHide)

	ropt := &report.Options{
		CumSort:      cfg.Sort == "cum",
		CallTree:     cfg.CallTree,
		DropNegative: cfg.DropNegative,

		CompactLabels: cfg.CompactLabels,
		Ratio:         1 / cfg.DivideBy,

		NodeCount:    cfg.NodeCount,
		NodeFraction: cfg.NodeFraction,
		EdgeFraction: cfg.EdgeFraction,

		ActiveFilters: filters,
		NumLabelUnits: numLabelUnits,

		SampleValue:       value,
		SampleMeanDivisor: meanDiv,
		SampleType:        stype,
		SampleUnit:        sample.Unit,

		OutputUnit: cfg.Unit,

		SourcePath: cfg.SourcePath,
		TrimPath:   cfg.TrimPath,

		IntelSyntax: cfg.IntelSyntax,
	}

	if len(p.Mapping) > 0 && p.Mapping[0].File != "" {
		ropt.Title = filepath.Base(p.Mapping[0].File)
	}

	return ropt, nil
}

// identifyNumLabelUnits returns a map of numeric label keys to the units
// associated with those keys.
func identifyNumLabelUnits(p *profile.Profile, ui plugin.UI) map[string]string {
	numLabelUnits, ignoredUnits := p.NumLabelUnits()

	// Print errors for tags with multiple units associated with
	// a single key.
	for k, units := range ignoredUnits {
		ui.PrintErr(fmt.Sprintf("For tag %s used unit %s, also encountered unit(s) %s", k, numLabelUnits[k], strings.Join(units, ", ")))
	}
	return numLabelUnits
}

type sampleValueFunc func([]int64) int64

// sampleFormat returns a function to extract values out of a profile.Sample,
// and the type/units of those values.
func sampleFormat(p *profile.Profile, sampleIndex string, mean bool) (value, meanDiv sampleValueFunc, v *profile.ValueType, err error) {
	if len(p.SampleType) == 0 {
		return nil, nil, nil, fmt.Errorf("profile has no samples")
	}
	index, err := p.SampleIndexByName(sampleIndex)
	if err != nil {
		return nil, nil, nil, err
	}
	value = valueExtractor(index)
	if mean {
		meanDiv = valueExtractor(0)
	}
	v = p.SampleType[index]
	return
}

func valueExtractor(ix int) sampleValueFunc {
	return func(v []int64) int64 {
		return v[ix]
	}
}

// profileCopier can be used to obtain a fresh copy of a profile.
// It is useful since reporting code may mutate the profile handed to it.
type profileCopier []byte

func makeProfileCopier(src *profile.Profile) profileCopier {
	// Pre-serialize the profile. We will deserialize every time a fresh copy is needed.
	var buf bytes.Buffer
	src.WriteUncompressed(&buf)
	return profileCopier(buf.Bytes())
}

// newCopy returns a new copy of the profile.
func (c profileCopier) newCopy() *profile.Profile {
	p, err := profile.ParseUncompressed([]byte(c))
	if err != nil {
		panic(err)
	}
	return p
}
