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
	"math/rand"
	"strings"
	"testing"

	"github.com/google/pprof/internal/plugin"
	"github.com/google/pprof/internal/proftest"
	"github.com/google/pprof/internal/report"
	"github.com/google/pprof/profile"
)

func TestShell(t *testing.T) {
	p := &profile.Profile{}
	generateReportWrapper = checkValue
	defer func() { generateReportWrapper = generateReport }()

	// Use test commands and variables to exercise interactive processing
	var savedCommands commands
	savedCommands, pprofCommands = pprofCommands, testCommands
	defer func() { pprofCommands = savedCommands }()

	savedVariables := pprofVariables
	defer func() { pprofVariables = savedVariables }()

	// Random interleave of independent scripts
	pprofVariables = testVariables(savedVariables)
	o := setDefaults(nil)
	o.UI = newUI(t, interleave(script, 0))
	if err := interactive(p, o); err != nil {
		t.Error("first attempt:", err)
	}
	// Random interleave of independent scripts
	pprofVariables = testVariables(savedVariables)
	o.UI = newUI(t, interleave(script, 1))
	if err := interactive(p, o); err != nil {
		t.Error("second attempt:", err)
	}

	// Random interleave of independent scripts with shortcuts
	pprofVariables = testVariables(savedVariables)
	var scScript []string
	pprofShortcuts, scScript = makeShortcuts(interleave(script, 2), 1)
	o.UI = newUI(t, scScript)
	if err := interactive(p, o); err != nil {
		t.Error("first shortcut attempt:", err)
	}

	// Random interleave of independent scripts with shortcuts
	pprofVariables = testVariables(savedVariables)
	pprofShortcuts, scScript = makeShortcuts(interleave(script, 1), 2)
	o.UI = newUI(t, scScript)
	if err := interactive(p, o); err != nil {
		t.Error("second shortcut attempt:", err)
	}

	// Group with invalid value
	pprofVariables = testVariables(savedVariables)
	ui := &proftest.TestUI{
		T:       t,
		Input:   []string{"cumulative=this"},
		AllowRx: `Unrecognized value for cumulative: "this". Use one of cum, flat`,
	}
	o.UI = ui
	if err := interactive(p, o); err != nil {
		t.Error("invalid group value:", err)
	}
	// Confirm error message written out once.
	if ui.NumAllowRxMatches != 1 {
		t.Errorf("want error message to be printed 1 time, got %v", ui.NumAllowRxMatches)
	}
	// Verify propagation of IO errors
	pprofVariables = testVariables(savedVariables)
	o.UI = newUI(t, []string{"**error**"})
	if err := interactive(p, o); err == nil {
		t.Error("expected IO error, got nil")
	}

}

var testCommands = commands{
	"check": &command{report.Raw, nil, nil, true, "", ""},
}

func testVariables(base variables) variables {
	v := base.makeCopy()

	v["b"] = &variable{boolKind, "f", "", ""}
	v["bb"] = &variable{boolKind, "f", "", ""}
	v["i"] = &variable{intKind, "0", "", ""}
	v["ii"] = &variable{intKind, "0", "", ""}
	v["f"] = &variable{floatKind, "0", "", ""}
	v["ff"] = &variable{floatKind, "0", "", ""}
	v["s"] = &variable{stringKind, "", "", ""}
	v["ss"] = &variable{stringKind, "", "", ""}

	v["ta"] = &variable{boolKind, "f", "radio", ""}
	v["tb"] = &variable{boolKind, "f", "radio", ""}
	v["tc"] = &variable{boolKind, "t", "radio", ""}

	return v
}

// script contains sequences of commands to be executed for testing. Commands
// are split by semicolon and interleaved randomly, so they must be
// independent from each other.
var script = []string{
	"bb=true;bb=false;check bb=false;bb=yes;check bb=true",
	"b=1;check b=true;b=n;check b=false",
	"i=-1;i=-2;check i=-2;i=999999;check i=999999",
	"check ii=0;ii=-1;check ii=-1;ii=100;check ii=100",
	"f=-1;f=-2.5;check f=-2.5;f=0.0001;check f=0.0001",
	"check ff=0;ff=-1.01;check ff=-1.01;ff=100;check ff=100",
	"s=one;s=two;check s=two",
	"ss=tree;check ss=tree;ss=;check ss;ss=forest;check ss=forest",
	"ta=true;check ta=true;check tb=false;check tc=false;tb=1;check tb=true;check ta=false;check tc=false;tc=yes;check tb=false;check ta=false;check tc=true",
}

func makeShortcuts(input []string, seed int) (shortcuts, []string) {
	rand.Seed(int64(seed))

	s := shortcuts{}
	var output, chunk []string
	for _, l := range input {
		chunk = append(chunk, l)
		switch rand.Intn(3) {
		case 0:
			// Create a macro for commands in 'chunk'.
			macro := fmt.Sprintf("alias%d", len(s))
			s[macro] = chunk
			output = append(output, macro)
			chunk = nil
		case 1:
			// Append commands in 'chunk' by themselves.
			output = append(output, chunk...)
			chunk = nil
		case 2:
			// Accumulate commands into 'chunk'
		}
	}
	output = append(output, chunk...)
	return s, output
}

func newUI(t *testing.T, input []string) plugin.UI {
	return &proftest.TestUI{
		T:     t,
		Input: input,
	}
}

func checkValue(p *profile.Profile, cmd []string, vars variables, o *plugin.Options) error {
	if len(cmd) != 2 {
		return fmt.Errorf("expected len(cmd)==2, got %v", cmd)
	}

	input := cmd[1]
	args := strings.SplitN(input, "=", 2)
	if len(args) == 0 {
		return fmt.Errorf("unexpected empty input")
	}
	name, value := args[0], ""
	if len(args) == 2 {
		value = args[1]
	}

	gotv := vars[name]
	if gotv == nil {
		return fmt.Errorf("Could not find variable named %s", name)
	}

	if got := gotv.stringValue(); got != value {
		return fmt.Errorf("Variable %s, want %s, got %s", name, value, got)
	}
	return nil
}

func interleave(input []string, seed int) []string {
	var inputs [][]string
	for _, s := range input {
		inputs = append(inputs, strings.Split(s, ";"))
	}
	rand.Seed(int64(seed))
	var output []string
	for len(inputs) > 0 {
		next := rand.Intn(len(inputs))
		output = append(output, inputs[next][0])
		if tail := inputs[next][1:]; len(tail) > 0 {
			inputs[next] = tail
		} else {
			inputs = append(inputs[:next], inputs[next+1:]...)
		}
	}
	return output
}

func TestInteractiveCommands(t *testing.T) {
	type interactiveTestcase struct {
		input string
		want  map[string]string
	}

	testcases := []interactiveTestcase{
		{
			"top 10 --cum focus1 -ignore focus2",
			map[string]string{
				"functions": "true",
				"nodecount": "10",
				"cum":       "true",
				"focus":     "focus1|focus2",
				"ignore":    "ignore",
			},
		},
		{
			"top10 --cum focus1 -ignore focus2",
			map[string]string{
				"functions": "true",
				"nodecount": "10",
				"cum":       "true",
				"focus":     "focus1|focus2",
				"ignore":    "ignore",
			},
		},
		{
			"dot",
			map[string]string{
				"functions": "true",
				"nodecount": "80",
				"cum":       "false",
			},
		},
		{
			"tags   -ignore1 -ignore2 focus1 >out",
			map[string]string{
				"functions": "true",
				"nodecount": "80",
				"cum":       "false",
				"output":    "out",
				"tagfocus":  "focus1",
				"tagignore": "ignore1|ignore2",
			},
		},
		{
			"weblist  find -test",
			map[string]string{
				"functions":        "false",
				"addressnoinlines": "true",
				"nodecount":        "0",
				"cum":              "false",
				"flat":             "true",
				"ignore":           "test",
			},
		},
		{
			"callgrind   fun -ignore  >out",
			map[string]string{
				"functions": "false",
				"addresses": "true",
				"nodecount": "0",
				"cum":       "false",
				"flat":      "true",
				"output":    "out",
			},
		},
		{
			"999",
			nil, // Error
		},
	}

	for _, tc := range testcases {
		cmd, vars, err := parseCommandLine(strings.Fields(tc.input))
		if tc.want == nil && err != nil {
			// Error expected
			continue
		}
		if err != nil {
			t.Errorf("failed on %q: %v", tc.input, err)
			continue
		}

		// Get report output format
		c := pprofCommands[cmd[0]]
		if c == nil {
			t.Errorf("unexpected nil command")
		}
		vars = applyCommandOverrides(cmd[0], c.format, vars)

		for n, want := range tc.want {
			if got := vars[n].stringValue(); got != want {
				t.Errorf("failed on %q, cmd=%q, %s got %s, want %s", tc.input, cmd, n, got, want)
			}
		}
	}
}
