// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package driver

import (
	"fmt"
	"io"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"cmd/pprof/internal/commands"
	"cmd/pprof/internal/plugin"
	"internal/pprof/profile"
)

var profileFunctionNames = []string{}

// functionCompleter replaces provided substring with a function
// name retrieved from a profile if a single match exists. Otherwise,
// it returns unchanged substring. It defaults to no-op if the profile
// is not specified.
func functionCompleter(substring string) string {
	found := ""
	for _, fName := range profileFunctionNames {
		if strings.Contains(fName, substring) {
			if found != "" {
				return substring
			}
			found = fName
		}
	}
	if found != "" {
		return found
	}
	return substring
}

// updateAutoComplete enhances autocompletion with information that can be
// retrieved from the profile
func updateAutoComplete(p *profile.Profile) {
	profileFunctionNames = nil // remove function names retrieved previously
	for _, fn := range p.Function {
		profileFunctionNames = append(profileFunctionNames, fn.Name)
	}
}

// splitCommand splits the command line input into tokens separated by
// spaces. Takes care to separate commands of the form 'top10' into
// two tokens: 'top' and '10'
func splitCommand(input string) []string {
	fields := strings.Fields(input)
	if num := strings.IndexAny(fields[0], "0123456789"); num != -1 {
		inputNumber := fields[0][num:]
		fields[0] = fields[0][:num]
		fields = append([]string{fields[0], inputNumber}, fields[1:]...)
	}
	return fields
}

// interactive displays a prompt and reads commands for profile
// manipulation/visualization.
func interactive(p *profile.Profile, obj plugin.ObjTool, ui plugin.UI, f *flags) error {
	updateAutoComplete(p)

	// Enter command processing loop.
	ui.Print("Entering interactive mode (type \"help\" for commands)")
	ui.SetAutoComplete(commands.NewCompleter(f.commands))

	for {
		input, err := readCommand(p, ui, f)
		if err != nil {
			if err != io.EOF {
				return err
			}
			if input == "" {
				return nil
			}
		}
		// Process simple commands.
		switch input {
		case "":
			continue
		case ":":
			f.flagFocus = newString("")
			f.flagIgnore = newString("")
			f.flagTagFocus = newString("")
			f.flagTagIgnore = newString("")
			f.flagHide = newString("")
			continue
		}

		fields := splitCommand(input)
		// Process report generation commands.
		if _, ok := f.commands[fields[0]]; ok {
			if err := generateReport(p, fields, obj, ui, f); err != nil {
				if err == io.EOF {
					return nil
				}
				ui.PrintErr(err)
			}
			continue
		}

		switch cmd := fields[0]; cmd {
		case "help":
			commandHelp(fields, ui, f)
			continue
		case "exit", "quit":
			return nil
		}

		// Process option settings.
		if of, err := optFlags(p, input, f); err == nil {
			f = of
		} else {
			ui.PrintErr("Error: ", err.Error())
		}
	}
}

func generateReport(p *profile.Profile, cmd []string, obj plugin.ObjTool, ui plugin.UI, f *flags) error {
	prof := p.Copy()

	cf, err := cmdFlags(prof, cmd, ui, f)
	if err != nil {
		return err
	}

	return generate(true, prof, obj, ui, cf)
}

// validateRegex checks if a string is a valid regular expression.
func validateRegex(v string) error {
	_, err := regexp.Compile(v)
	return err
}

// readCommand prompts for and reads the next command.
func readCommand(p *profile.Profile, ui plugin.UI, f *flags) (string, error) {
	//ui.Print("Options:\n", f.String(p))
	s, err := ui.ReadLine()
	return strings.TrimSpace(s), err
}

func commandHelp(_ []string, ui plugin.UI, f *flags) error {
	help := `
 Commands:
   cmd [n] [--cum] [focus_regex]* [-ignore_regex]*
       Produce a text report with the top n entries.
       Include samples matching focus_regex, and exclude ignore_regex.
       Add --cum to sort using cumulative data.
       Available commands:
`
	var commands []string
	for name, cmd := range f.commands {
		commands = append(commands, fmt.Sprintf("         %-12s %s", name, cmd.Usage))
	}
	sort.Strings(commands)

	help = help + strings.Join(commands, "\n") + `
   peek func_regex
       Display callers and callees of functions matching func_regex.

   dot [n] [focus_regex]* [-ignore_regex]* [>file]
       Produce an annotated callgraph with the top n entries.
       Include samples matching focus_regex, and exclude ignore_regex.
       For other outputs, replace dot with:
       - Graphic formats: dot, svg, pdf, ps, gif, png (use > to name output file)
       - Graph viewer:    gv, web, evince, eog

   callgrind [n] [focus_regex]* [-ignore_regex]* [>file]
       Produce a file in callgrind-compatible format.
       Include samples matching focus_regex, and exclude ignore_regex.

   weblist func_regex [-ignore_regex]*
       Show annotated source with interspersed assembly in a web browser.

   list func_regex [-ignore_regex]*
       Print source for routines matching func_regex, and exclude ignore_regex.

   disasm func_regex [-ignore_regex]*
       Disassemble routines matching func_regex, and exclude ignore_regex.

   tags tag_regex [-ignore_regex]*
       List tags with key:value matching tag_regex and exclude ignore_regex.

   quit/exit/^D
 	     Exit pprof.

   option=value
       The following options can be set individually:
           cum/flat:           Sort entries based on cumulative or flat data
           call_tree:          Build context-sensitive call trees
           nodecount:          Max number of entries to display
           nodefraction:       Min frequency ratio of nodes to display
           edgefraction:       Min frequency ratio of edges to display
           focus/ignore:       Regexp to include/exclude samples by name/file
           tagfocus/tagignore: Regexp or value range to filter samples by tag
                               eg "1mb", "1mb:2mb", ":64kb"

           functions:          Level of aggregation for sample data
           files:
           lines:
           addresses:

           unit:               Measurement unit to use on reports

           Sample value selection by index:
            sample_index:      Index of sample value to display
            mean:              Average sample value over first value

           Sample value selection by name:
            alloc_space        for heap profiles
            alloc_objects
            inuse_space
            inuse_objects

            total_delay        for contention profiles
            mean_delay
            contentions

   :   Clear focus/ignore/hide/tagfocus/tagignore`

	ui.Print(help)
	return nil
}

// cmdFlags parses the options of an interactive command and returns
// an updated flags object.
func cmdFlags(prof *profile.Profile, input []string, ui plugin.UI, f *flags) (*flags, error) {
	cf := *f

	var focus, ignore string
	output := *cf.flagOutput
	nodeCount := *cf.flagNodeCount
	cmd := input[0]

	// Update output flags based on parameters.
	tokens := input[1:]
	for p := 0; p < len(tokens); p++ {
		t := tokens[p]
		if t == "" {
			continue
		}
		if c, err := strconv.ParseInt(t, 10, 32); err == nil {
			nodeCount = int(c)
			continue
		}
		switch t[0] {
		case '>':
			if len(t) > 1 {
				output = t[1:]
				continue
			}
			// find next token
			for p++; p < len(tokens); p++ {
				if tokens[p] != "" {
					output = tokens[p]
					break
				}
			}
		case '-':
			if t == "--cum" || t == "-cum" {
				cf.flagCum = newBool(true)
				continue
			}
			ignore = catRegex(ignore, t[1:])
		default:
			focus = catRegex(focus, t)
		}
	}

	pcmd, ok := f.commands[cmd]
	if !ok {
		return nil, fmt.Errorf("Unexpected parse failure: %v", input)
	}
	// Reset flags
	cf.flagCommands = make(map[string]*bool)
	cf.flagParamCommands = make(map[string]*string)

	if !pcmd.HasParam {
		cf.flagCommands[cmd] = newBool(true)

		switch cmd {
		case "tags":
			cf.flagTagFocus = newString(focus)
			cf.flagTagIgnore = newString(ignore)
		default:
			cf.flagFocus = newString(catRegex(*cf.flagFocus, focus))
			cf.flagIgnore = newString(catRegex(*cf.flagIgnore, ignore))
		}
	} else {
		if focus == "" {
			focus = "."
		}
		cf.flagParamCommands[cmd] = newString(focus)
		cf.flagIgnore = newString(catRegex(*cf.flagIgnore, ignore))
	}

	if nodeCount < 0 {
		switch cmd {
		case "text", "top":
			// Default text/top to 10 nodes on interactive mode
			nodeCount = 10
		default:
			nodeCount = 80
		}
	}

	cf.flagNodeCount = newInt(nodeCount)
	cf.flagOutput = newString(output)

	// Do regular flags processing
	if err := processFlags(prof, ui, &cf); err != nil {
		cf.usage(ui)
		return nil, err
	}

	return &cf, nil
}

func catRegex(a, b string) string {
	if a == "" {
		return b
	}
	if b == "" {
		return a
	}
	return a + "|" + b
}

// optFlags parses an interactive option setting and returns
// an updated flags object.
func optFlags(p *profile.Profile, input string, f *flags) (*flags, error) {
	inputs := strings.SplitN(input, "=", 2)
	option := strings.ToLower(strings.TrimSpace(inputs[0]))
	var value string
	if len(inputs) == 2 {
		value = strings.TrimSpace(inputs[1])
	}

	of := *f

	var err error
	var bv bool
	var uv uint64
	var fv float64

	switch option {
	case "cum":
		if bv, err = parseBool(value); err != nil {
			return nil, err
		}
		of.flagCum = newBool(bv)
	case "flat":
		if bv, err = parseBool(value); err != nil {
			return nil, err
		}
		of.flagCum = newBool(!bv)
	case "call_tree":
		if bv, err = parseBool(value); err != nil {
			return nil, err
		}
		of.flagCallTree = newBool(bv)
	case "unit":
		of.flagDisplayUnit = newString(value)
	case "sample_index":
		if uv, err = strconv.ParseUint(value, 10, 32); err != nil {
			return nil, err
		}
		if ix := int(uv); ix < 0 || ix >= len(p.SampleType) {
			return nil, fmt.Errorf("sample_index out of range [0..%d]", len(p.SampleType)-1)
		}
		of.flagSampleIndex = newInt(int(uv))
	case "mean":
		if bv, err = parseBool(value); err != nil {
			return nil, err
		}
		of.flagMean = newBool(bv)
	case "nodecount":
		if uv, err = strconv.ParseUint(value, 10, 32); err != nil {
			return nil, err
		}
		of.flagNodeCount = newInt(int(uv))
	case "nodefraction":
		if fv, err = strconv.ParseFloat(value, 64); err != nil {
			return nil, err
		}
		of.flagNodeFraction = newFloat64(fv)
	case "edgefraction":
		if fv, err = strconv.ParseFloat(value, 64); err != nil {
			return nil, err
		}
		of.flagEdgeFraction = newFloat64(fv)
	case "focus":
		if err = validateRegex(value); err != nil {
			return nil, err
		}
		of.flagFocus = newString(value)
	case "ignore":
		if err = validateRegex(value); err != nil {
			return nil, err
		}
		of.flagIgnore = newString(value)
	case "tagfocus":
		if err = validateRegex(value); err != nil {
			return nil, err
		}
		of.flagTagFocus = newString(value)
	case "tagignore":
		if err = validateRegex(value); err != nil {
			return nil, err
		}
		of.flagTagIgnore = newString(value)
	case "hide":
		if err = validateRegex(value); err != nil {
			return nil, err
		}
		of.flagHide = newString(value)
	case "addresses", "files", "lines", "functions":
		if bv, err = parseBool(value); err != nil {
			return nil, err
		}
		if !bv {
			return nil, fmt.Errorf("select one of addresses/files/lines/functions")
		}
		setGranularityToggle(option, &of)
	default:
		if ix := findSampleIndex(p, "", option); ix >= 0 {
			of.flagSampleIndex = newInt(ix)
		} else if ix := findSampleIndex(p, "total_", option); ix >= 0 {
			of.flagSampleIndex = newInt(ix)
			of.flagMean = newBool(false)
		} else if ix := findSampleIndex(p, "mean_", option); ix >= 1 {
			of.flagSampleIndex = newInt(ix)
			of.flagMean = newBool(true)
		} else {
			return nil, fmt.Errorf("unrecognized command: %s", input)
		}
	}
	return &of, nil
}

// parseBool parses a string as a boolean value.
func parseBool(v string) (bool, error) {
	switch strings.ToLower(v) {
	case "true", "t", "yes", "y", "1", "":
		return true, nil
	case "false", "f", "no", "n", "0":
		return false, nil
	}
	return false, fmt.Errorf(`illegal input "%s" for bool value`, v)
}

func findSampleIndex(p *profile.Profile, prefix, sampleType string) int {
	if !strings.HasPrefix(sampleType, prefix) {
		return -1
	}
	sampleType = strings.TrimPrefix(sampleType, prefix)
	for i, r := range p.SampleType {
		if r.Type == sampleType {
			return i
		}
	}
	return -1
}

// setGranularityToggle manages the set of granularity options. These
// operate as a toggle; turning one on turns the others off.
func setGranularityToggle(o string, fl *flags) {
	t, f := newBool(true), newBool(false)
	fl.flagFunctions = f
	fl.flagFiles = f
	fl.flagLines = f
	fl.flagAddresses = f
	switch o {
	case "functions":
		fl.flagFunctions = t
	case "files":
		fl.flagFiles = t
	case "lines":
		fl.flagLines = t
	case "addresses":
		fl.flagAddresses = t
	default:
		panic(fmt.Errorf("unexpected option %s", o))
	}
}
