// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package analysisflags defines helpers for processing flags of
// analysis driver tools.
package analysisflags

import (
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"

	"golang.org/x/tools/go/analysis"
)

// Parse creates a flag for each of the analyzer's flags,
// including (in multi mode) a flag named after the analyzer,
// parses the flags, then filters and returns the list of
// analyzers enabled by flags.
func Parse(analyzers []*analysis.Analyzer, multi bool) []*analysis.Analyzer {
	// Connect each analysis flag to the command line as -analysis.flag.
	type analysisFlag struct {
		Name  string
		Bool  bool
		Usage string
	}
	var analysisFlags []analysisFlag

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
			analysisFlags = append(analysisFlags, analysisFlag{a.Name, true, enableUsage})
		}

		a.Flags.VisitAll(func(f *flag.Flag) {
			if !multi && flag.Lookup(f.Name) != nil {
				log.Printf("%s flag -%s would conflict with driver; skipping", a.Name, f.Name)
				return
			}

			name := prefix + f.Name
			flag.Var(f.Value, name, f.Usage)

			var isBool bool
			if b, ok := f.Value.(interface{ IsBoolFlag() bool }); ok {
				isBool = b.IsBoolFlag()
			}
			analysisFlags = append(analysisFlags, analysisFlag{name, isBool, f.Usage})
		})
	}

	// standard flags: -flags, -V.
	printflags := flag.Bool("flags", false, "print analyzer flags in JSON")
	addVersionFlag()

	// Add shims for legacy vet flags.
	for old, new := range vetLegacyFlags {
		newFlag := flag.Lookup(new)
		if newFlag != nil && flag.Lookup(old) == nil {
			flag.Var(newFlag.Value, old, "deprecated alias for -"+new)
		}
	}

	flag.Parse() // (ExitOnError)

	// -flags: print flags so that go vet knows which ones are legitimate.
	if *printflags {
		data, err := json.MarshalIndent(analysisFlags, "", "\t")
		if err != nil {
			log.Fatal(err)
		}
		os.Stdout.Write(data)
		os.Exit(0)
	}

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

	return analyzers
}

// addVersionFlag registers a -V flag that, if set,
// prints the executable version and exits 0.
//
// It is a variable not a function to permit easy
// overriding in the copy vendored in $GOROOT/src/cmd/vet:
//
// func init() { addVersionFlag = objabi.AddVersionFlag }
var addVersionFlag = func() {
	flag.Var(versionFlag{}, "V", "print version and exit")
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
