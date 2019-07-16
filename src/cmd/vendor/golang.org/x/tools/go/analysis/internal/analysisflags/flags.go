// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package analysisflags defines helpers for processing flags of
// analysis driver tools.
package analysisflags

import (
	"crypto/sha256"
	"encoding/gob"
	"encoding/json"
	"flag"
	"fmt"
	"go/token"
	"io"
	"io/ioutil"
	"log"
	"os"
	"strconv"
	"strings"

	"golang.org/x/tools/go/analysis"
)

// flags common to all {single,multi,unit}checkers.
var (
	JSON    = false // -json
	Context = -1    // -c=N: if N>0, display offending line plus N lines of context
)

// Parse creates a flag for each of the analyzer's flags,
// including (in multi mode) a flag named after the analyzer,
// parses the flags, then filters and returns the list of
// analyzers enabled by flags.
//
// The result is intended to be passed to unitchecker.Run or checker.Run.
// Use in unitchecker.Run will gob.Register all fact types for the returned
// graph of analyzers but of course not the ones only reachable from
// dropped analyzers. To avoid inconsistency about which gob types are
// registered from run to run, Parse itself gob.Registers all the facts
// only reachable from dropped analyzers.
// This is not a particularly elegant API, but this is an internal package.
func Parse(analyzers []*analysis.Analyzer, multi bool) []*analysis.Analyzer {
	// Connect each analysis flag to the command line as -analysis.flag.
	enabled := make(map[*analysis.Analyzer]*triState)
	for _, a := range analyzers {
		var prefix string

		// Add -NAME flag to enable it.
		if multi {
			prefix = a.Name + "."

			enable := new(triState)
			enableUsage := "enable " + a.Name + " analysis"
			flag.Var(enable, a.Name, enableUsage)
			enabled[a] = enable
		}

		a.Flags.VisitAll(func(f *flag.Flag) {
			if !multi && flag.Lookup(f.Name) != nil {
				log.Printf("%s flag -%s would conflict with driver; skipping", a.Name, f.Name)
				return
			}

			name := prefix + f.Name
			flag.Var(f.Value, name, f.Usage)
		})
	}

	// standard flags: -flags, -V.
	printflags := flag.Bool("flags", false, "print analyzer flags in JSON")
	addVersionFlag()

	// flags common to all checkers
	flag.BoolVar(&JSON, "json", JSON, "emit JSON output")
	flag.IntVar(&Context, "c", Context, `display offending line with this many lines of context`)

	// Add shims for legacy vet flags to enable existing
	// scripts that run vet to continue to work.
	_ = flag.Bool("source", false, "no effect (deprecated)")
	_ = flag.Bool("v", false, "no effect (deprecated)")
	_ = flag.Bool("all", false, "no effect (deprecated)")
	_ = flag.String("tags", "", "no effect (deprecated)")
	for old, new := range vetLegacyFlags {
		newFlag := flag.Lookup(new)
		if newFlag != nil && flag.Lookup(old) == nil {
			flag.Var(newFlag.Value, old, "deprecated alias for -"+new)
		}
	}

	flag.Parse() // (ExitOnError)

	// -flags: print flags so that go vet knows which ones are legitimate.
	if *printflags {
		printFlags()
		os.Exit(0)
	}

	everything := expand(analyzers)

	// If any -NAME flag is true,  run only those analyzers. Otherwise,
	// if any -NAME flag is false, run all but those analyzers.
	if multi {
		var hasTrue, hasFalse bool
		for _, ts := range enabled {
			switch *ts {
			case setTrue:
				hasTrue = true
			case setFalse:
				hasFalse = true
			}
		}

		var keep []*analysis.Analyzer
		if hasTrue {
			for _, a := range analyzers {
				if *enabled[a] == setTrue {
					keep = append(keep, a)
				}
			}
			analyzers = keep
		} else if hasFalse {
			for _, a := range analyzers {
				if *enabled[a] != setFalse {
					keep = append(keep, a)
				}
			}
			analyzers = keep
		}
	}

	// Register fact types of skipped analyzers
	// in case we encounter them in imported files.
	kept := expand(analyzers)
	for a := range everything {
		if !kept[a] {
			for _, f := range a.FactTypes {
				gob.Register(f)
			}
		}
	}

	return analyzers
}

func expand(analyzers []*analysis.Analyzer) map[*analysis.Analyzer]bool {
	seen := make(map[*analysis.Analyzer]bool)
	var visitAll func([]*analysis.Analyzer)
	visitAll = func(analyzers []*analysis.Analyzer) {
		for _, a := range analyzers {
			if !seen[a] {
				seen[a] = true
				visitAll(a.Requires)
			}
		}
	}
	visitAll(analyzers)
	return seen
}

func printFlags() {
	type jsonFlag struct {
		Name  string
		Bool  bool
		Usage string
	}
	var flags []jsonFlag = nil
	flag.VisitAll(func(f *flag.Flag) {
		// Don't report {single,multi}checker debugging
		// flags as these have no effect on unitchecker
		// (as invoked by 'go vet').
		switch f.Name {
		case "debug", "cpuprofile", "memprofile", "trace":
			return
		}

		b, ok := f.Value.(interface{ IsBoolFlag() bool })
		isBool := ok && b.IsBoolFlag()
		flags = append(flags, jsonFlag{f.Name, isBool, f.Usage})
	})
	data, err := json.MarshalIndent(flags, "", "\t")
	if err != nil {
		log.Fatal(err)
	}
	os.Stdout.Write(data)
}

// addVersionFlag registers a -V flag that, if set,
// prints the executable version and exits 0.
//
// If the -V flag already exists — for example, because it was already
// registered by a call to cmd/internal/objabi.AddVersionFlag — then
// addVersionFlag does nothing.
func addVersionFlag() {
	if flag.Lookup("V") == nil {
		flag.Var(versionFlag{}, "V", "print version and exit")
	}
}

// versionFlag minimally complies with the -V protocol required by "go vet".
type versionFlag struct{}

func (versionFlag) IsBoolFlag() bool { return true }
func (versionFlag) Get() interface{} { return nil }
func (versionFlag) String() string   { return "" }
func (versionFlag) Set(s string) error {
	if s != "full" {
		log.Fatalf("unsupported flag value: -V=%s", s)
	}

	// This replicates the miminal subset of
	// cmd/internal/objabi.AddVersionFlag, which is private to the
	// go tool yet forms part of our command-line interface.
	// TODO(adonovan): clarify the contract.

	// Print the tool version so the build system can track changes.
	// Formats:
	//   $progname version devel ... buildID=...
	//   $progname version go1.9.1
	progname := os.Args[0]
	f, err := os.Open(progname)
	if err != nil {
		log.Fatal(err)
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		log.Fatal(err)
	}
	f.Close()
	fmt.Printf("%s version devel comments-go-here buildID=%02x\n",
		progname, string(h.Sum(nil)))
	os.Exit(0)
	return nil
}

// A triState is a boolean that knows whether
// it has been set to either true or false.
// It is used to identify whether a flag appears;
// the standard boolean flag cannot
// distinguish missing from unset.
// It also satisfies flag.Value.
type triState int

const (
	unset triState = iota
	setTrue
	setFalse
)

func triStateFlag(name string, value triState, usage string) *triState {
	flag.Var(&value, name, usage)
	return &value
}

// triState implements flag.Value, flag.Getter, and flag.boolFlag.
// They work like boolean flags: we can say vet -printf as well as vet -printf=true
func (ts *triState) Get() interface{} {
	return *ts == setTrue
}

func (ts triState) isTrue() bool {
	return ts == setTrue
}

func (ts *triState) Set(value string) error {
	b, err := strconv.ParseBool(value)
	if err != nil {
		// This error message looks poor but package "flag" adds
		// "invalid boolean value %q for -NAME: %s"
		return fmt.Errorf("want true or false")
	}
	if b {
		*ts = setTrue
	} else {
		*ts = setFalse
	}
	return nil
}

func (ts *triState) String() string {
	switch *ts {
	case unset:
		return "true"
	case setTrue:
		return "true"
	case setFalse:
		return "false"
	}
	panic("not reached")
}

func (ts triState) IsBoolFlag() bool {
	return true
}

// Legacy flag support

// vetLegacyFlags maps flags used by legacy vet to their corresponding
// new names. The old names will continue to work.
var vetLegacyFlags = map[string]string{
	// Analyzer name changes
	"bool":       "bools",
	"buildtags":  "buildtag",
	"methods":    "stdmethods",
	"rangeloops": "loopclosure",

	// Analyzer flags
	"compositewhitelist":  "composites.whitelist",
	"printfuncs":          "printf.funcs",
	"shadowstrict":        "shadow.strict",
	"unusedfuncs":         "unusedresult.funcs",
	"unusedstringmethods": "unusedresult.stringmethods",
}

// ---- output helpers common to all drivers ----

// PrintPlain prints a diagnostic in plain text form,
// with context specified by the -c flag.
func PrintPlain(fset *token.FileSet, diag analysis.Diagnostic) {
	posn := fset.Position(diag.Pos)
	fmt.Fprintf(os.Stderr, "%s: %s\n", posn, diag.Message)

	// -c=N: show offending line plus N lines of context.
	if Context >= 0 {
		posn := fset.Position(diag.Pos)
		end := fset.Position(diag.End)
		if !end.IsValid() {
			end = posn
		}
		data, _ := ioutil.ReadFile(posn.Filename)
		lines := strings.Split(string(data), "\n")
		for i := posn.Line - Context; i <= end.Line+Context; i++ {
			if 1 <= i && i <= len(lines) {
				fmt.Fprintf(os.Stderr, "%d\t%s\n", i, lines[i-1])
			}
		}
	}
}

// A JSONTree is a mapping from package ID to analysis name to result.
// Each result is either a jsonError or a list of jsonDiagnostic.
type JSONTree map[string]map[string]interface{}

// Add adds the result of analysis 'name' on package 'id'.
// The result is either a list of diagnostics or an error.
func (tree JSONTree) Add(fset *token.FileSet, id, name string, diags []analysis.Diagnostic, err error) {
	var v interface{}
	if err != nil {
		type jsonError struct {
			Err string `json:"error"`
		}
		v = jsonError{err.Error()}
	} else if len(diags) > 0 {
		type jsonDiagnostic struct {
			Category string `json:"category,omitempty"`
			Posn     string `json:"posn"`
			Message  string `json:"message"`
		}
		var diagnostics []jsonDiagnostic
		// TODO(matloob): Should the JSON diagnostics contain ranges?
		// If so, how should they be formatted?
		for _, f := range diags {
			diagnostics = append(diagnostics, jsonDiagnostic{
				Category: f.Category,
				Posn:     fset.Position(f.Pos).String(),
				Message:  f.Message,
			})
		}
		v = diagnostics
	}
	if v != nil {
		m, ok := tree[id]
		if !ok {
			m = make(map[string]interface{})
			tree[id] = m
		}
		m[name] = v
	}
}

func (tree JSONTree) Print() {
	data, err := json.MarshalIndent(tree, "", "\t")
	if err != nil {
		log.Panicf("internal error: JSON marshalling failed: %v", err)
	}
	fmt.Printf("%s\n", data)
}
