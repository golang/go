// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Debug arguments, set by -d flag.

package base

import (
	"fmt"
	"log"
	"os"
	"reflect"
	"strconv"
	"strings"
)

// Debug holds the parsed debugging configuration values.
var Debug DebugFlags

// DebugFlags defines the debugging configuration values (see var Debug).
// Each struct field is a different value, named for the lower-case of the field name.
// Each field must be an int or string and must have a `help` struct tag.
//
// The -d option takes a comma-separated list of settings.
// Each setting is name=value; for ints, name is short for name=1.
type DebugFlags struct {
	Append               int    `help:"print information about append compilation"`
	Checkptr             int    `help:"instrument unsafe pointer conversions"`
	Closure              int    `help:"print information about closure compilation"`
	DclStack             int    `help:"run internal dclstack check"`
	Defer                int    `help:"print information about defer compilation"`
	DisableNil           int    `help:"disable nil checks"`
	DumpPtrs             int    `help:"show Node pointers values in dump output"`
	DwarfInl             int    `help:"print information about DWARF inlined function creation"`
	Export               int    `help:"print export data"`
	GCProg               int    `help:"print dump of GC programs"`
	InlFuncsWithClosures int    `help:"allow functions with closures to be inlined"`
	Libfuzzer            int    `help:"enable coverage instrumentation for libfuzzer"`
	LocationLists        int    `help:"print information about DWARF location list creation"`
	Nil                  int    `help:"print information about nil checks"`
	NoOpenDefer          int    `help:"disable open-coded defers"`
	PCTab                string `help:"print named pc-value table"`
	Panic                int    `help:"show all compiler panics"`
	Slice                int    `help:"print information about slice compilation"`
	SoftFloat            int    `help:"force compiler to emit soft-float code"`
	TypeAssert           int    `help:"print information about type assertion inlining"`
	TypecheckInl         int    `help:"eager typechecking of inline function bodies"`
	WB                   int    `help:"print information about write barriers"`
	ABIWrap              int    `help:"print information about ABI wrapper generation"`

	any bool // set when any of the values have been set
}

// Any reports whether any of the debug flags have been set.
func (d *DebugFlags) Any() bool { return d.any }

type debugField struct {
	name string
	help string
	val  interface{} // *int or *string
}

var debugTab []debugField

func init() {
	v := reflect.ValueOf(&Debug).Elem()
	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if f.Name == "any" {
			continue
		}
		name := strings.ToLower(f.Name)
		help := f.Tag.Get("help")
		if help == "" {
			panic(fmt.Sprintf("base.Debug.%s is missing help text", f.Name))
		}
		ptr := v.Field(i).Addr().Interface()
		switch ptr.(type) {
		default:
			panic(fmt.Sprintf("base.Debug.%s has invalid type %v (must be int or string)", f.Name, f.Type))
		case *int, *string:
			// ok
		}
		debugTab = append(debugTab, debugField{name, help, ptr})
	}
}

// DebugSSA is called to set a -d ssa/... option.
// If nil, those options are reported as invalid options.
// If DebugSSA returns a non-empty string, that text is reported as a compiler error.
var DebugSSA func(phase, flag string, val int, valString string) string

// parseDebug parses the -d debug string argument.
func parseDebug(debugstr string) {
	// parse -d argument
	if debugstr == "" {
		return
	}
	Debug.any = true
Split:
	for _, name := range strings.Split(debugstr, ",") {
		if name == "" {
			continue
		}
		// display help about the -d option itself and quit
		if name == "help" {
			fmt.Print(debugHelpHeader)
			maxLen := len("ssa/help")
			for _, t := range debugTab {
				if len(t.name) > maxLen {
					maxLen = len(t.name)
				}
			}
			for _, t := range debugTab {
				fmt.Printf("\t%-*s\t%s\n", maxLen, t.name, t.help)
			}
			// ssa options have their own help
			fmt.Printf("\t%-*s\t%s\n", maxLen, "ssa/help", "print help about SSA debugging")
			fmt.Print(debugHelpFooter)
			os.Exit(0)
		}
		val, valstring, haveInt := 1, "", true
		if i := strings.IndexAny(name, "=:"); i >= 0 {
			var err error
			name, valstring = name[:i], name[i+1:]
			val, err = strconv.Atoi(valstring)
			if err != nil {
				val, haveInt = 1, false
			}
		}
		for _, t := range debugTab {
			if t.name != name {
				continue
			}
			switch vp := t.val.(type) {
			case nil:
				// Ignore
			case *string:
				*vp = valstring
			case *int:
				if !haveInt {
					log.Fatalf("invalid debug value %v", name)
				}
				*vp = val
			default:
				panic("bad debugtab type")
			}
			continue Split
		}
		// special case for ssa for now
		if DebugSSA != nil && strings.HasPrefix(name, "ssa/") {
			// expect form ssa/phase/flag
			// e.g. -d=ssa/generic_cse/time
			// _ in phase name also matches space
			phase := name[4:]
			flag := "debug" // default flag is debug
			if i := strings.Index(phase, "/"); i >= 0 {
				flag = phase[i+1:]
				phase = phase[:i]
			}
			err := DebugSSA(phase, flag, val, valstring)
			if err != "" {
				log.Fatalf(err)
			}
			continue Split
		}
		log.Fatalf("unknown debug key -d %s\n", name)
	}
}

const debugHelpHeader = `usage: -d arg[,arg]* and arg is <key>[=<value>]

<key> is one of:

`

const debugHelpFooter = `
<value> is key-specific.

Key "checkptr" supports values:
	"0": instrumentation disabled
	"1": conversions involving unsafe.Pointer are instrumented
	"2": conversions to unsafe.Pointer force heap allocation

Key "pctab" supports values:
	"pctospadj", "pctofile", "pctoline", "pctoinline", "pctopcdata"
`
