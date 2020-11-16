// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"cmd/compile/internal/ssa"
	"cmd/internal/objabi"
)

// Debug arguments.
// These can be specified with the -d flag, as in "-d nil"
// to set the debug_checknil variable.
// Multiple options can be comma-separated.
// Each option accepts an optional argument, as in "gcprog=2"
var debugtab = []struct {
	name string
	help string
	val  interface{} // must be *int or *string
}{
	{"append", "print information about append compilation", &Debug.Append},
	{"checkptr", "instrument unsafe pointer conversions", &Debug.Checkptr},
	{"closure", "print information about closure compilation", &Debug.Closure},
	{"compilelater", "compile functions as late as possible", &Debug.CompileLater},
	{"disablenil", "disable nil checks", &Debug.DisableNil},
	{"dclstack", "run internal dclstack check", &Debug.DclStack},
	{"dumpptrs", "show Node pointer values in Dump/dumplist output", &Debug.DumpPtrs},
	{"gcprog", "print dump of GC programs", &Debug.GCProg},
	{"libfuzzer", "coverage instrumentation for libfuzzer", &Debug.Libfuzzer},
	{"nil", "print information about nil checks", &Debug.Nil},
	{"panic", "do not hide any compiler panic", &Debug.Panic},
	{"slice", "print information about slice compilation", &Debug.Slice},
	{"typeassert", "print information about type assertion inlining", &Debug.TypeAssert},
	{"wb", "print information about write barriers", &Debug.WB},
	{"export", "print export data", &Debug.Export},
	{"pctab", "print named pc-value table", &Debug.PCTab},
	{"locationlists", "print information about DWARF location list creation", &Debug.LocationLists},
	{"typecheckinl", "eager typechecking of inline function bodies", &Debug.TypecheckInl},
	{"dwarfinl", "print information about DWARF inlined function creation", &Debug.DwarfInl},
	{"softfloat", "force compiler to emit soft-float code", &Debug.SoftFloat},
	{"defer", "print information about defer compilation", &Debug.Defer},
	{"fieldtrack", "enable fieldtracking", &objabi.Fieldtrack_enabled},
}

var Debug struct {
	Append        int
	Checkptr      int
	Closure       int
	CompileLater  int
	DisableNil    int
	DclStack      int
	GCProg        int
	Libfuzzer     int
	Nil           int
	Panic         int
	Slice         int
	TypeAssert    int
	WB            int
	Export        int
	PCTab         string
	LocationLists int
	TypecheckInl  int
	DwarfInl      int
	SoftFloat     int
	Defer         int
	DumpPtrs      int
}

func parseDebug() {
	// parse -d argument
	if Flag.LowerD != "" {
	Split:
		for _, name := range strings.Split(Flag.LowerD, ",") {
			if name == "" {
				continue
			}
			// display help about the -d option itself and quit
			if name == "help" {
				fmt.Print(debugHelpHeader)
				maxLen := len("ssa/help")
				for _, t := range debugtab {
					if len(t.name) > maxLen {
						maxLen = len(t.name)
					}
				}
				for _, t := range debugtab {
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
			for _, t := range debugtab {
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
			if strings.HasPrefix(name, "ssa/") {
				// expect form ssa/phase/flag
				// e.g. -d=ssa/generic_cse/time
				// _ in phase name also matches space
				phase := name[4:]
				flag := "debug" // default flag is debug
				if i := strings.Index(phase, "/"); i >= 0 {
					flag = phase[i+1:]
					phase = phase[:i]
				}
				err := ssa.PhaseOption(phase, flag, val, valstring)
				if err != "" {
					log.Fatalf(err)
				}
				continue Split
			}
			log.Fatalf("unknown debug key -d %s\n", name)
		}
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
