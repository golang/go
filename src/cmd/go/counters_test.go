// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"flag"
	"go/build"
	"internal/diff"
	"os"
	"slices"
	"strings"
	"testing"
)

var update = flag.Bool("update", false, "if true update testdata/counternames.txt")

func TestCounterNamesUpToDate(t *testing.T) {
	if !*update {
		t.Parallel()
	}

	var counters []string
	// -C is a special case because it's handled by handleChdirFlag rather than
	// standard flag processing with FlagSets.
	// go/subcommand:unknown is also a special case: it's used when the subcommand
	// doesn't match any of the known commands.
	counters = append(counters, "go/flag:C", "go/subcommand:unknown")
	counters = append(counters, flagscounters("go/flag:", *flag.CommandLine)...)

	// Add help (without any arguments) as a special case. cmdcounters adds go help <cmd>
	// for all subcommands, but it's also valid to invoke go help without any arguments.
	counters = append(counters, "go/subcommand:help")
	for _, cmd := range base.Go.Commands {
		cmdcounters, err := cmdcounters(nil, cmd)
		if err != nil {
			t.Fatal(err)
		}
		counters = append(counters, cmdcounters...)
	}

	counters = append(counters, base.RegisteredCounterNames()...)
	for _, c := range counters {
		const counterPrefix = "go/"
		if !strings.HasPrefix(c, counterPrefix) {
			t.Fatalf("registered counter %q does not start with %q", c, counterPrefix)
		}
	}

	cstr := []byte(strings.Join(counters, "\n") + "\n")
	const counterNamesFile = "testdata/counters.txt"
	old, err := os.ReadFile(counterNamesFile)
	if err != nil {
		t.Fatalf("error reading %s: %v", counterNamesFile, err)
	}
	diff := diff.Diff(counterNamesFile, old, "generated counter names", cstr)
	if diff == nil {
		t.Logf("%s is up to date.", counterNamesFile)
		return
	}

	if *update {
		if err := os.WriteFile(counterNamesFile, cstr, 0666); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %d bytes to %s", len(cstr), counterNamesFile)
		t.Logf("don't forget to file a proposal to update the list of collected counters")
	} else {
		t.Logf("\n%s", diff)
		t.Errorf("%s is stale. To update, run 'go generate cmd/go'.", counterNamesFile)
	}
}

func flagscounters(prefix string, flagSet flag.FlagSet) []string {
	var counters []string
	flagSet.VisitAll(func(f *flag.Flag) {
		counters = append(counters, prefix+f.Name)
	})
	return counters
}

func cmdcounters(previous []string, cmd *base.Command) ([]string, error) {
	const subcommandPrefix = "go/subcommand:"
	const flagPrefix = "go/flag:"
	var counters []string
	previousComponent := strings.Join(previous, "-")
	if len(previousComponent) > 0 {
		previousComponent += "-"
	}
	if cmd.Runnable() {
		if cmd.Name() == "tool" {
			// TODO(matloob): Do we expect the same tools to be present on all
			// platforms/configurations? Should we only run this on certain
			// platforms?
			tools, err := toolNames()
			if err != nil {
				return nil, err
			}
			for _, t := range tools {
				counters = append(counters, subcommandPrefix+previousComponent+cmd.Name()+"-"+t)
			}
			counters = append(counters, subcommandPrefix+previousComponent+cmd.Name()+"-unknown")
		}
		counters = append(counters, subcommandPrefix+previousComponent+cmd.Name())
	}
	counters = append(counters, flagscounters(flagPrefix+previousComponent+cmd.Name()+"-", cmd.Flag)...)
	if len(previous) != 0 {
		counters = append(counters, subcommandPrefix+previousComponent+"help-"+cmd.Name())
	}
	counters = append(counters, subcommandPrefix+"help-"+previousComponent+cmd.Name())

	for _, subcmd := range cmd.Commands {
		subcmdcounters, err := cmdcounters(append(slices.Clone(previous), cmd.Name()), subcmd)
		if err != nil {
			return nil, err
		}
		counters = append(counters, subcmdcounters...)
	}
	return counters, nil
}

// toolNames returns the list of basenames of executables in the tool dir.
func toolNames() ([]string, error) {
	entries, err := os.ReadDir(build.ToolDir)
	if err != nil {
		return nil, err
	}
	var names []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := strings.TrimSuffix(e.Name(), cfg.ToolExeSuffix())
		names = append(names, name)
	}
	return names, nil
}
