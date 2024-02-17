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
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/google/pprof/internal/plugin"
	"github.com/google/pprof/internal/report"
)

// commands describes the commands accepted by pprof.
type commands map[string]*command

// command describes the actions for a pprof command. Includes a
// function for command-line completion, the report format to use
// during report generation, any postprocessing functions, and whether
// the command expects a regexp parameter (typically a function name).
type command struct {
	format      int           // report format to generate
	postProcess PostProcessor // postprocessing to run on report
	visualizer  PostProcessor // display output using some callback
	hasParam    bool          // collect a parameter from the CLI
	description string        // single-line description text saying what the command does
	usage       string        // multi-line help text saying how the command is used
}

// help returns a help string for a command.
func (c *command) help(name string) string {
	message := c.description + "\n"
	if c.usage != "" {
		message += "  Usage:\n"
		lines := strings.Split(c.usage, "\n")
		for _, line := range lines {
			message += fmt.Sprintf("    %s\n", line)
		}
	}
	return message + "\n"
}

// AddCommand adds an additional command to the set of commands
// accepted by pprof. This enables extensions to add new commands for
// specialized visualization formats. If the command specified already
// exists, it is overwritten.
func AddCommand(cmd string, format int, post PostProcessor, desc, usage string) {
	pprofCommands[cmd] = &command{format, post, nil, false, desc, usage}
}

// SetVariableDefault sets the default value for a pprof
// variable. This enables extensions to set their own defaults.
func SetVariableDefault(variable, value string) {
	configure(variable, value)
}

// PostProcessor is a function that applies post-processing to the report output
type PostProcessor func(input io.Reader, output io.Writer, ui plugin.UI) error

// interactiveMode is true if pprof is running on interactive mode, reading
// commands from its shell.
var interactiveMode = false

// pprofCommands are the report generation commands recognized by pprof.
var pprofCommands = commands{
	// Commands that require no post-processing.
	"comments": {report.Comments, nil, nil, false, "Output all profile comments", ""},
	"disasm":   {report.Dis, nil, nil, true, "Output assembly listings annotated with samples", listHelp("disasm", true)},
	"dot":      {report.Dot, nil, nil, false, "Outputs a graph in DOT format", reportHelp("dot", false, true)},
	"list":     {report.List, nil, nil, true, "Output annotated source for functions matching regexp", listHelp("list", false)},
	"peek":     {report.Tree, nil, nil, true, "Output callers/callees of functions matching regexp", "peek func_regex\nDisplay callers and callees of functions matching func_regex."},
	"raw":      {report.Raw, nil, nil, false, "Outputs a text representation of the raw profile", ""},
	"tags":     {report.Tags, nil, nil, false, "Outputs all tags in the profile", "tags [tag_regex]* [-ignore_regex]* [>file]\nList tags with key:value matching tag_regex and exclude ignore_regex."},
	"text":     {report.Text, nil, nil, false, "Outputs top entries in text form", reportHelp("text", true, true)},
	"top":      {report.Text, nil, nil, false, "Outputs top entries in text form", reportHelp("top", true, true)},
	"traces":   {report.Traces, nil, nil, false, "Outputs all profile samples in text form", ""},
	"tree":     {report.Tree, nil, nil, false, "Outputs a text rendering of call graph", reportHelp("tree", true, true)},

	// Save binary formats to a file
	"callgrind": {report.Callgrind, nil, awayFromTTY("callgraph.out"), false, "Outputs a graph in callgrind format", reportHelp("callgrind", false, true)},
	"proto":     {report.Proto, nil, awayFromTTY("pb.gz"), false, "Outputs the profile in compressed protobuf format", ""},
	"topproto":  {report.TopProto, nil, awayFromTTY("pb.gz"), false, "Outputs top entries in compressed protobuf format", ""},

	// Generate report in DOT format and postprocess with dot
	"gif": {report.Dot, invokeDot("gif"), awayFromTTY("gif"), false, "Outputs a graph image in GIF format", reportHelp("gif", false, true)},
	"pdf": {report.Dot, invokeDot("pdf"), awayFromTTY("pdf"), false, "Outputs a graph in PDF format", reportHelp("pdf", false, true)},
	"png": {report.Dot, invokeDot("png"), awayFromTTY("png"), false, "Outputs a graph image in PNG format", reportHelp("png", false, true)},
	"ps":  {report.Dot, invokeDot("ps"), awayFromTTY("ps"), false, "Outputs a graph in PS format", reportHelp("ps", false, true)},

	// Save SVG output into a file
	"svg": {report.Dot, massageDotSVG(), awayFromTTY("svg"), false, "Outputs a graph in SVG format", reportHelp("svg", false, true)},

	// Visualize postprocessed dot output
	"eog":    {report.Dot, invokeDot("svg"), invokeVisualizer("svg", []string{"eog"}), false, "Visualize graph through eog", reportHelp("eog", false, false)},
	"evince": {report.Dot, invokeDot("pdf"), invokeVisualizer("pdf", []string{"evince"}), false, "Visualize graph through evince", reportHelp("evince", false, false)},
	"gv":     {report.Dot, invokeDot("ps"), invokeVisualizer("ps", []string{"gv --noantialias"}), false, "Visualize graph through gv", reportHelp("gv", false, false)},
	"web":    {report.Dot, massageDotSVG(), invokeVisualizer("svg", browsers()), false, "Visualize graph through web browser", reportHelp("web", false, false)},

	// Visualize callgrind output
	"kcachegrind": {report.Callgrind, nil, invokeVisualizer("grind", kcachegrind), false, "Visualize report in KCachegrind", reportHelp("kcachegrind", false, false)},

	// Visualize HTML directly generated by report.
	"weblist": {report.WebList, nil, invokeVisualizer("html", browsers()), true, "Display annotated source in a web browser", listHelp("weblist", false)},
}

// configHelp contains help text per configuration parameter.
var configHelp = map[string]string{
	// Filename for file-based output formats, stdout by default.
	"output": helpText("Output filename for file-based outputs"),

	// Comparisons.
	"drop_negative": helpText(
		"Ignore negative differences",
		"Do not show any locations with values <0."),

	// Graph handling options.
	"call_tree": helpText(
		"Create a context-sensitive call tree",
		"Treat locations reached through different paths as separate."),

	// Display options.
	"relative_percentages": helpText(
		"Show percentages relative to focused subgraph",
		"If unset, percentages are relative to full graph before focusing",
		"to facilitate comparison with original graph."),
	"unit": helpText(
		"Measurement units to display",
		"Scale the sample values to this unit.",
		"For time-based profiles, use seconds, milliseconds, nanoseconds, etc.",
		"For memory profiles, use megabytes, kilobytes, bytes, etc.",
		"Using auto will scale each value independently to the most natural unit."),
	"compact_labels": "Show minimal headers",
	"source_path":    "Search path for source files",
	"trim_path":      "Path to trim from source paths before search",
	"intel_syntax": helpText(
		"Show assembly in Intel syntax",
		"Only applicable to commands `disasm` and `weblist`"),

	// Filtering options
	"nodecount": helpText(
		"Max number of nodes to show",
		"Uses heuristics to limit the number of locations to be displayed.",
		"On graphs, dotted edges represent paths through nodes that have been removed."),
	"nodefraction": "Hide nodes below <f>*total",
	"edgefraction": "Hide edges below <f>*total",
	"trim": helpText(
		"Honor nodefraction/edgefraction/nodecount defaults",
		"Set to false to get the full profile, without any trimming."),
	"focus": helpText(
		"Restricts to samples going through a node matching regexp",
		"Discard samples that do not include a node matching this regexp.",
		"Matching includes the function name, filename or object name."),
	"ignore": helpText(
		"Skips paths going through any nodes matching regexp",
		"If set, discard samples that include a node matching this regexp.",
		"Matching includes the function name, filename or object name."),
	"prune_from": helpText(
		"Drops any functions below the matched frame.",
		"If set, any frames matching the specified regexp and any frames",
		"below it will be dropped from each sample."),
	"hide": helpText(
		"Skips nodes matching regexp",
		"Discard nodes that match this location.",
		"Other nodes from samples that include this location will be shown.",
		"Matching includes the function name, filename or object name."),
	"show": helpText(
		"Only show nodes matching regexp",
		"If set, only show nodes that match this location.",
		"Matching includes the function name, filename or object name."),
	"show_from": helpText(
		"Drops functions above the highest matched frame.",
		"If set, all frames above the highest match are dropped from every sample.",
		"Matching includes the function name, filename or object name."),
	"tagroot": helpText(
		"Adds pseudo stack frames for labels key/value pairs at the callstack root.",
		"A comma-separated list of label keys.",
		"The first key creates frames at the new root."),
	"tagleaf": helpText(
		"Adds pseudo stack frames for labels key/value pairs at the callstack leaf.",
		"A comma-separated list of label keys.",
		"The last key creates frames at the new leaf."),
	"tagfocus": helpText(
		"Restricts to samples with tags in range or matched by regexp",
		"Use name=value syntax to limit the matching to a specific tag.",
		"Numeric tag filter examples: 1kb, 1kb:10kb, memory=32mb:",
		"String tag filter examples: foo, foo.*bar, mytag=foo.*bar"),
	"tagignore": helpText(
		"Discard samples with tags in range or matched by regexp",
		"Use name=value syntax to limit the matching to a specific tag.",
		"Numeric tag filter examples: 1kb, 1kb:10kb, memory=32mb:",
		"String tag filter examples: foo, foo.*bar, mytag=foo.*bar"),
	"tagshow": helpText(
		"Only consider tags matching this regexp",
		"Discard tags that do not match this regexp"),
	"taghide": helpText(
		"Skip tags matching this regexp",
		"Discard tags that match this regexp"),
	// Heap profile options
	"divide_by": helpText(
		"Ratio to divide all samples before visualization",
		"Divide all samples values by a constant, eg the number of processors or jobs."),
	"mean": helpText(
		"Average sample value over first value (count)",
		"For memory profiles, report average memory per allocation.",
		"For time-based profiles, report average time per event."),
	"sample_index": helpText(
		"Sample value to report (0-based index or name)",
		"Profiles contain multiple values per sample.",
		"Use sample_index=i to select the ith value (starting at 0)."),
	"normalize": helpText(
		"Scales profile based on the base profile."),

	// Data sorting criteria
	"flat": helpText("Sort entries based on own weight"),
	"cum":  helpText("Sort entries based on cumulative weight"),

	// Output granularity
	"functions": helpText(
		"Aggregate at the function level.",
		"Ignores the filename where the function was defined."),
	"filefunctions": helpText(
		"Aggregate at the function level.",
		"Takes into account the filename where the function was defined."),
	"files": "Aggregate at the file level.",
	"lines": "Aggregate at the source code line level.",
	"addresses": helpText(
		"Aggregate at the address level.",
		"Includes functions' addresses in the output."),
	"noinlines": helpText(
		"Ignore inlines.",
		"Attributes inlined functions to their first out-of-line caller."),
	"showcolumns": helpText(
		"Show column numbers at the source code line level."),
}

func helpText(s ...string) string {
	return strings.Join(s, "\n") + "\n"
}

// usage returns a string describing the pprof commands and configuration
// options.  if commandLine is set, the output reflect cli usage.
func usage(commandLine bool) string {
	var prefix string
	if commandLine {
		prefix = "-"
	}
	fmtHelp := func(c, d string) string {
		return fmt.Sprintf("    %-16s %s", c, strings.SplitN(d, "\n", 2)[0])
	}

	var commands []string
	for name, cmd := range pprofCommands {
		commands = append(commands, fmtHelp(prefix+name, cmd.description))
	}
	sort.Strings(commands)

	var help string
	if commandLine {
		help = "  Output formats (select at most one):\n"
	} else {
		help = "  Commands:\n"
		commands = append(commands, fmtHelp("o/options", "List options and their current values"))
		commands = append(commands, fmtHelp("q/quit/exit/^D", "Exit pprof"))
	}

	help = help + strings.Join(commands, "\n") + "\n\n" +
		"  Options:\n"

	// Print help for configuration options after sorting them.
	// Collect choices for multi-choice options print them together.
	var variables []string
	var radioStrings []string
	for _, f := range configFields {
		if len(f.choices) == 0 {
			variables = append(variables, fmtHelp(prefix+f.name, configHelp[f.name]))
			continue
		}
		// Format help for for this group.
		s := []string{fmtHelp(f.name, "")}
		for _, choice := range f.choices {
			s = append(s, "  "+fmtHelp(prefix+choice, configHelp[choice]))
		}
		radioStrings = append(radioStrings, strings.Join(s, "\n"))
	}
	sort.Strings(variables)
	sort.Strings(radioStrings)
	return help + strings.Join(variables, "\n") + "\n\n" +
		"  Option groups (only set one per group):\n" +
		strings.Join(radioStrings, "\n")
}

func reportHelp(c string, cum, redirect bool) string {
	h := []string{
		c + " [n] [focus_regex]* [-ignore_regex]*",
		"Include up to n samples",
		"Include samples matching focus_regex, and exclude ignore_regex.",
	}
	if cum {
		h[0] += " [-cum]"
		h = append(h, "-cum sorts the output by cumulative weight")
	}
	if redirect {
		h[0] += " >f"
		h = append(h, "Optionally save the report on the file f")
	}
	return strings.Join(h, "\n")
}

func listHelp(c string, redirect bool) string {
	h := []string{
		c + "<func_regex|address> [-focus_regex]* [-ignore_regex]*",
		"Include functions matching func_regex, or including the address specified.",
		"Include samples matching focus_regex, and exclude ignore_regex.",
	}
	if redirect {
		h[0] += " >f"
		h = append(h, "Optionally save the report on the file f")
	}
	return strings.Join(h, "\n")
}

// browsers returns a list of commands to attempt for web visualization.
func browsers() []string {
	var cmds []string
	if userBrowser := os.Getenv("BROWSER"); userBrowser != "" {
		cmds = append(cmds, userBrowser)
	}
	switch runtime.GOOS {
	case "darwin":
		cmds = append(cmds, "/usr/bin/open")
	case "windows":
		cmds = append(cmds, "cmd /c start")
	default:
		// Commands opening browsers are prioritized over xdg-open, so browser()
		// command can be used on linux to open the .svg file generated by the -web
		// command (the .svg file includes embedded javascript so is best viewed in
		// a browser).
		cmds = append(cmds, []string{"chrome", "google-chrome", "chromium", "firefox", "sensible-browser"}...)
		if os.Getenv("DISPLAY") != "" {
			// xdg-open is only for use in a desktop environment.
			cmds = append(cmds, "xdg-open")
		}
	}
	return cmds
}

var kcachegrind = []string{"kcachegrind"}

// awayFromTTY saves the output in a file if it would otherwise go to
// the terminal screen. This is used to avoid dumping binary data on
// the screen.
func awayFromTTY(format string) PostProcessor {
	return func(input io.Reader, output io.Writer, ui plugin.UI) error {
		if output == os.Stdout && (ui.IsTerminal() || interactiveMode) {
			tempFile, err := newTempFile("", "profile", "."+format)
			if err != nil {
				return err
			}
			ui.PrintErr("Generating report in ", tempFile.Name())
			output = tempFile
		}
		_, err := io.Copy(output, input)
		return err
	}
}

func invokeDot(format string) PostProcessor {
	return func(input io.Reader, output io.Writer, ui plugin.UI) error {
		cmd := exec.Command("dot", "-T"+format)
		cmd.Stdin, cmd.Stdout, cmd.Stderr = input, output, os.Stderr
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("failed to execute dot. Is Graphviz installed? Error: %v", err)
		}
		return nil
	}
}

// massageDotSVG invokes the dot tool to generate an SVG image and alters
// the image to have panning capabilities when viewed in a browser.
func massageDotSVG() PostProcessor {
	generateSVG := invokeDot("svg")
	return func(input io.Reader, output io.Writer, ui plugin.UI) error {
		baseSVG := new(bytes.Buffer)
		if err := generateSVG(input, baseSVG, ui); err != nil {
			return err
		}
		_, err := output.Write([]byte(massageSVG(baseSVG.String())))
		return err
	}
}

func invokeVisualizer(suffix string, visualizers []string) PostProcessor {
	return func(input io.Reader, output io.Writer, ui plugin.UI) error {
		tempFile, err := newTempFile(os.TempDir(), "pprof", "."+suffix)
		if err != nil {
			return err
		}
		deferDeleteTempFile(tempFile.Name())
		if _, err := io.Copy(tempFile, input); err != nil {
			return err
		}
		tempFile.Close()
		// Try visualizers until one is successful
		for _, v := range visualizers {
			// Separate command and arguments for exec.Command.
			args := strings.Split(v, " ")
			if len(args) == 0 {
				continue
			}
			viewer := exec.Command(args[0], append(args[1:], tempFile.Name())...)
			viewer.Stderr = os.Stderr
			if err = viewer.Start(); err == nil {
				// Wait for a second so that the visualizer has a chance to
				// open the input file. This needs to be done even if we're
				// waiting for the visualizer as it can be just a wrapper that
				// spawns a browser tab and returns right away.
				defer func(t <-chan time.Time) {
					<-t
				}(time.After(time.Second))
				// On interactive mode, let the visualizer run in the background
				// so other commands can be issued.
				if !interactiveMode {
					return viewer.Wait()
				}
				return nil
			}
		}
		return err
	}
}

// stringToBool is a custom parser for bools. We avoid using strconv.ParseBool
// to remain compatible with old pprof behavior (e.g., treating "" as true).
func stringToBool(s string) (bool, error) {
	switch strings.ToLower(s) {
	case "true", "t", "yes", "y", "1", "":
		return true, nil
	case "false", "f", "no", "n", "0":
		return false, nil
	default:
		return false, fmt.Errorf(`illegal value "%s" for bool variable`, s)
	}
}
