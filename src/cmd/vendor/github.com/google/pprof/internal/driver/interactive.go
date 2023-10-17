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
	"io"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/google/pprof/internal/plugin"
	"github.com/google/pprof/internal/report"
	"github.com/google/pprof/profile"
)

var commentStart = "//:" // Sentinel for comments on options
var tailDigitsRE = regexp.MustCompile("[0-9]+$")

// interactive starts a shell to read pprof commands.
func interactive(p *profile.Profile, o *plugin.Options) error {
	// Enter command processing loop.
	o.UI.SetAutoComplete(newCompleter(functionNames(p)))
	configure("compact_labels", "true")
	configHelp["sample_index"] += fmt.Sprintf("Or use sample_index=name, with name in %v.\n", sampleTypes(p))

	// Do not wait for the visualizer to complete, to allow multiple
	// graphs to be visualized simultaneously.
	interactiveMode = true
	shortcuts := profileShortcuts(p)

	copier := makeProfileCopier(p)
	greetings(p, o.UI)
	for {
		input, err := o.UI.ReadLine("(pprof) ")
		if err != nil {
			if err != io.EOF {
				return err
			}
			if input == "" {
				return nil
			}
		}

		for _, input := range shortcuts.expand(input) {
			// Process assignments of the form variable=value
			if s := strings.SplitN(input, "=", 2); len(s) > 0 {
				name := strings.TrimSpace(s[0])
				var value string
				if len(s) == 2 {
					value = s[1]
					if comment := strings.LastIndex(value, commentStart); comment != -1 {
						value = value[:comment]
					}
					value = strings.TrimSpace(value)
				}
				if isConfigurable(name) {
					// All non-bool options require inputs
					if len(s) == 1 && !isBoolConfig(name) {
						o.UI.PrintErr(fmt.Errorf("please specify a value, e.g. %s=<val>", name))
						continue
					}
					if name == "sample_index" {
						// Error check sample_index=xxx to ensure xxx is a valid sample type.
						index, err := p.SampleIndexByName(value)
						if err != nil {
							o.UI.PrintErr(err)
							continue
						}
						if index < 0 || index >= len(p.SampleType) {
							o.UI.PrintErr(fmt.Errorf("invalid sample_index %q", value))
							continue
						}
						value = p.SampleType[index].Type
					}
					if err := configure(name, value); err != nil {
						o.UI.PrintErr(err)
					}
					continue
				}
			}

			tokens := strings.Fields(input)
			if len(tokens) == 0 {
				continue
			}

			switch tokens[0] {
			case "o", "options":
				printCurrentOptions(p, o.UI)
				continue
			case "exit", "quit", "q":
				return nil
			case "help":
				commandHelp(strings.Join(tokens[1:], " "), o.UI)
				continue
			}

			args, cfg, err := parseCommandLine(tokens)
			if err == nil {
				err = generateReportWrapper(copier.newCopy(), args, cfg, o)
			}

			if err != nil {
				o.UI.PrintErr(err)
			}
		}
	}
}

var generateReportWrapper = generateReport // For testing purposes.

// greetings prints a brief welcome and some overall profile
// information before accepting interactive commands.
func greetings(p *profile.Profile, ui plugin.UI) {
	numLabelUnits := identifyNumLabelUnits(p, ui)
	ropt, err := reportOptions(p, numLabelUnits, currentConfig())
	if err == nil {
		rpt := report.New(p, ropt)
		ui.Print(strings.Join(report.ProfileLabels(rpt), "\n"))
		if rpt.Total() == 0 && len(p.SampleType) > 1 {
			ui.Print(`No samples were found with the default sample value type.`)
			ui.Print(`Try "sample_index" command to analyze different sample values.`, "\n")
		}
	}
	ui.Print(`Entering interactive mode (type "help" for commands, "o" for options)`)
}

// shortcuts represents composite commands that expand into a sequence
// of other commands.
type shortcuts map[string][]string

func (a shortcuts) expand(input string) []string {
	input = strings.TrimSpace(input)
	if a != nil {
		if r, ok := a[input]; ok {
			return r
		}
	}
	return []string{input}
}

var pprofShortcuts = shortcuts{
	":": []string{"focus=", "ignore=", "hide=", "tagfocus=", "tagignore="},
}

// profileShortcuts creates macros for convenience and backward compatibility.
func profileShortcuts(p *profile.Profile) shortcuts {
	s := pprofShortcuts
	// Add shortcuts for sample types
	for _, st := range p.SampleType {
		command := fmt.Sprintf("sample_index=%s", st.Type)
		s[st.Type] = []string{command}
		s["total_"+st.Type] = []string{"mean=0", command}
		s["mean_"+st.Type] = []string{"mean=1", command}
	}
	return s
}

func sampleTypes(p *profile.Profile) []string {
	types := make([]string, len(p.SampleType))
	for i, t := range p.SampleType {
		types[i] = t.Type
	}
	return types
}

func printCurrentOptions(p *profile.Profile, ui plugin.UI) {
	var args []string
	current := currentConfig()
	for _, f := range configFields {
		n := f.name
		v := current.get(f)
		comment := ""
		switch {
		case len(f.choices) > 0:
			values := append([]string{}, f.choices...)
			sort.Strings(values)
			comment = "[" + strings.Join(values, " | ") + "]"
		case n == "sample_index":
			st := sampleTypes(p)
			if v == "" {
				// Apply default (last sample index).
				v = st[len(st)-1]
			}
			// Add comments for all sample types in profile.
			comment = "[" + strings.Join(st, " | ") + "]"
		case n == "source_path":
			continue
		case n == "nodecount" && v == "-1":
			comment = "default"
		case v == "":
			// Add quotes for empty values.
			v = `""`
		}
		if comment != "" {
			comment = commentStart + " " + comment
		}
		args = append(args, fmt.Sprintf("  %-25s = %-20s %s", n, v, comment))
	}
	sort.Strings(args)
	ui.Print(strings.Join(args, "\n"))
}

// parseCommandLine parses a command and returns the pprof command to
// execute and the configuration to use for the report.
func parseCommandLine(input []string) ([]string, config, error) {
	cmd, args := input[:1], input[1:]
	name := cmd[0]

	c := pprofCommands[name]
	if c == nil {
		// Attempt splitting digits on abbreviated commands (eg top10)
		if d := tailDigitsRE.FindString(name); d != "" && d != name {
			name = name[:len(name)-len(d)]
			cmd[0], args = name, append([]string{d}, args...)
			c = pprofCommands[name]
		}
	}
	if c == nil {
		if _, ok := configHelp[name]; ok {
			value := "<val>"
			if len(args) > 0 {
				value = args[0]
			}
			return nil, config{}, fmt.Errorf("did you mean: %s=%s", name, value)
		}
		return nil, config{}, fmt.Errorf("unrecognized command: %q", name)
	}

	if c.hasParam {
		if len(args) == 0 {
			return nil, config{}, fmt.Errorf("command %s requires an argument", name)
		}
		cmd = append(cmd, args[0])
		args = args[1:]
	}

	// Copy config since options set in the command line should not persist.
	vcopy := currentConfig()

	var focus, ignore string
	for i := 0; i < len(args); i++ {
		t := args[i]
		if n, err := strconv.ParseInt(t, 10, 32); err == nil {
			vcopy.NodeCount = int(n)
			continue
		}
		switch t[0] {
		case '>':
			outputFile := t[1:]
			if outputFile == "" {
				i++
				if i >= len(args) {
					return nil, config{}, fmt.Errorf("unexpected end of line after >")
				}
				outputFile = args[i]
			}
			vcopy.Output = outputFile
		case '-':
			if t == "--cum" || t == "-cum" {
				vcopy.Sort = "cum"
				continue
			}
			ignore = catRegex(ignore, t[1:])
		default:
			focus = catRegex(focus, t)
		}
	}

	if name == "tags" {
		if focus != "" {
			vcopy.TagFocus = focus
		}
		if ignore != "" {
			vcopy.TagIgnore = ignore
		}
	} else {
		if focus != "" {
			vcopy.Focus = focus
		}
		if ignore != "" {
			vcopy.Ignore = ignore
		}
	}
	if vcopy.NodeCount == -1 && (name == "text" || name == "top") {
		vcopy.NodeCount = 10
	}

	return cmd, vcopy, nil
}

func catRegex(a, b string) string {
	if a != "" && b != "" {
		return a + "|" + b
	}
	return a + b
}

// commandHelp displays help and usage information for all Commands
// and Variables or a specific Command or Variable.
func commandHelp(args string, ui plugin.UI) {
	if args == "" {
		help := usage(false)
		help = help + `
  :   Clear focus/ignore/hide/tagfocus/tagignore

  type "help <cmd|option>" for more information
`

		ui.Print(help)
		return
	}

	if c := pprofCommands[args]; c != nil {
		ui.Print(c.help(args))
		return
	}

	if help, ok := configHelp[args]; ok {
		ui.Print(help + "\n")
		return
	}

	ui.PrintErr("Unknown command: " + args)
}

// newCompleter creates an autocompletion function for a set of commands.
func newCompleter(fns []string) func(string) string {
	return func(line string) string {
		switch tokens := strings.Fields(line); len(tokens) {
		case 0:
			// Nothing to complete
		case 1:
			// Single token -- complete command name
			if match := matchVariableOrCommand(tokens[0]); match != "" {
				return match
			}
		case 2:
			if tokens[0] == "help" {
				if match := matchVariableOrCommand(tokens[1]); match != "" {
					return tokens[0] + " " + match
				}
				return line
			}
			fallthrough
		default:
			// Multiple tokens -- complete using functions, except for tags
			if cmd := pprofCommands[tokens[0]]; cmd != nil && tokens[0] != "tags" {
				lastTokenIdx := len(tokens) - 1
				lastToken := tokens[lastTokenIdx]
				if strings.HasPrefix(lastToken, "-") {
					lastToken = "-" + functionCompleter(lastToken[1:], fns)
				} else {
					lastToken = functionCompleter(lastToken, fns)
				}
				return strings.Join(append(tokens[:lastTokenIdx], lastToken), " ")
			}
		}
		return line
	}
}

// matchVariableOrCommand attempts to match a string token to the prefix of a Command.
func matchVariableOrCommand(token string) string {
	token = strings.ToLower(token)
	var matches []string
	for cmd := range pprofCommands {
		if strings.HasPrefix(cmd, token) {
			matches = append(matches, cmd)
		}
	}
	matches = append(matches, completeConfig(token)...)
	if len(matches) == 1 {
		return matches[0]
	}
	return ""
}

// functionCompleter replaces provided substring with a function
// name retrieved from a profile if a single match exists. Otherwise,
// it returns unchanged substring. It defaults to no-op if the profile
// is not specified.
func functionCompleter(substring string, fns []string) string {
	found := ""
	for _, fName := range fns {
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

func functionNames(p *profile.Profile) []string {
	var fns []string
	for _, fn := range p.Function {
		fns = append(fns, fn.Name)
	}
	return fns
}
