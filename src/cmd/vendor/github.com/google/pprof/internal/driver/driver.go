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
		return generateReport(p, cmd, pprofVariables, o)
	}

	if src.HTTPHostport != "" {
		return serveWebInterface(src.HTTPHostport, p, o, src.HTTPDisableBrowser)
	}
	return interactive(p, o)
}

func generateRawReport(p *profile.Profile, cmd []string, vars variables, o *plugin.Options) (*command, *report.Report, error) {
	p = p.Copy() // Prevent modification to the incoming profile.

	// Identify units of numeric tags in profile.
	numLabelUnits := identifyNumLabelUnits(p, o.UI)

	// Get report output format
	c := pprofCommands[cmd[0]]
	if c == nil {
		panic("unexpected nil command")
	}

	vars = applyCommandOverrides(cmd[0], c.format, vars)

	// Delay focus after configuring report to get percentages on all samples.
	relative := vars["relative_percentages"].boolValue()
	if relative {
		if err := applyFocus(p, numLabelUnits, vars, o.UI); err != nil {
			return nil, nil, err
		}
	}
	ropt, err := reportOptions(p, numLabelUnits, vars)
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
		if err := applyFocus(p, numLabelUnits, vars, o.UI); err != nil {
			return nil, nil, err
		}
	}
	if err := aggregate(p, vars); err != nil {
		return nil, nil, err
	}

	return c, rpt, nil
}

func generateReport(p *profile.Profile, cmd []string, vars variables, o *plugin.Options) error {
	c, rpt, err := generateRawReport(p, cmd, vars, o)
	if err != nil {
		return err
	}

	// Generate the report.
	dst := new(bytes.Buffer)
	if err := report.Generate(dst, rpt, o.Obj); err != nil {
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
	output := vars["output"].value
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

func applyCommandOverrides(cmd string, outputFormat int, v variables) variables {
	// Some report types override the trim flag to false below. This is to make
	// sure the default heuristics of excluding insignificant nodes and edges
	// from the call graph do not apply. One example where it is important is
	// annotated source or disassembly listing. Those reports run on a specific
	// function (or functions), but the trimming is applied before the function
	// data is selected. So, with trimming enabled, the report could end up
	// showing no data if the specified function is "uninteresting" as far as the
	// trimming is concerned.
	trim := v["trim"].boolValue()

	switch cmd {
	case "disasm", "weblist":
		trim = false
		v.set("addresses", "t")
		// Force the 'noinlines' mode so that source locations for a given address
		// collapse and there is only one for the given address. Without this
		// cumulative metrics would be double-counted when annotating the assembly.
		// This is because the merge is done by address and in case of an inlined
		// stack each of the inlined entries is a separate callgraph node.
		v.set("noinlines", "t")
	case "peek":
		trim = false
	case "list":
		trim = false
		v.set("lines", "t")
		// Do not force 'noinlines' to be false so that specifying
		// "-list foo -noinlines" is supported and works as expected.
	case "text", "top", "topproto":
		if v["nodecount"].intValue() == -1 {
			v.set("nodecount", "0")
		}
	default:
		if v["nodecount"].intValue() == -1 {
			v.set("nodecount", "80")
		}
	}

	switch outputFormat {
	case report.Proto, report.Raw, report.Callgrind:
		trim = false
		v.set("addresses", "t")
		v.set("noinlines", "f")
	}

	if !trim {
		v.set("nodecount", "0")
		v.set("nodefraction", "0")
		v.set("edgefraction", "0")
	}
	return v
}

func aggregate(prof *profile.Profile, v variables) error {
	var function, filename, linenumber, address bool
	inlines := !v["noinlines"].boolValue()
	switch {
	case v["addresses"].boolValue():
		if inlines {
			return nil
		}
		function = true
		filename = true
		linenumber = true
		address = true
	case v["lines"].boolValue():
		function = true
		filename = true
		linenumber = true
	case v["files"].boolValue():
		filename = true
	case v["functions"].boolValue():
		function = true
	case v["filefunctions"].boolValue():
		function = true
		filename = true
	default:
		return fmt.Errorf("unexpected granularity")
	}
	return prof.Aggregate(inlines, function, filename, linenumber, address)
}

func reportOptions(p *profile.Profile, numLabelUnits map[string]string, vars variables) (*report.Options, error) {
	si, mean := vars["sample_index"].value, vars["mean"].boolValue()
	value, meanDiv, sample, err := sampleFormat(p, si, mean)
	if err != nil {
		return nil, err
	}

	stype := sample.Type
	if mean {
		stype = "mean_" + stype
	}

	if vars["divide_by"].floatValue() == 0 {
		return nil, fmt.Errorf("zero divisor specified")
	}

	var filters []string
	for _, k := range []string{"focus", "ignore", "hide", "show", "show_from", "tagfocus", "tagignore", "tagshow", "taghide"} {
		v := vars[k].value
		if v != "" {
			filters = append(filters, k+"="+v)
		}
	}

	ropt := &report.Options{
		CumSort:      vars["cum"].boolValue(),
		CallTree:     vars["call_tree"].boolValue(),
		DropNegative: vars["drop_negative"].boolValue(),

		CompactLabels: vars["compact_labels"].boolValue(),
		Ratio:         1 / vars["divide_by"].floatValue(),

		NodeCount:    vars["nodecount"].intValue(),
		NodeFraction: vars["nodefraction"].floatValue(),
		EdgeFraction: vars["edgefraction"].floatValue(),

		ActiveFilters: filters,
		NumLabelUnits: numLabelUnits,

		SampleValue:       value,
		SampleMeanDivisor: meanDiv,
		SampleType:        stype,
		SampleUnit:        sample.Unit,

		OutputUnit: vars["unit"].value,

		SourcePath: vars["source_path"].stringValue(),
		TrimPath:   vars["trim_path"].stringValue(),
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
