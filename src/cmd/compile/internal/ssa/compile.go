// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"cmd/internal/src"
)

type Compiler interface {
	Compile(f *Func, htmlWriter HTMLWriter)
	Passes() []Pass
}

type HTMLWriter interface {
	Enabled() bool
	FlushPhases()
	WritePhase(phase, title string)
	WriteColumn(phase, title, class, html string)
	DebugInfo(v func(*Value) string)
	TimeFormatting() time.Duration
	Close()
}

type Pass struct {
	Name     string
	Fn       func(*Func)
	Required bool
	Disabled bool
	Time     bool             // report time to run pass
	Mem      bool             // report mem stats to run pass
	Stats    int              // pass reports own "stats" (e.g., branches removed)
	Debug    int              // pass performs some debugging. =1 should be in error-testing-friendly Warnl format.
	Test     int              // pass-specific ad-hoc option, perhaps useful in development
	Dump     map[string]bool  // dump if function name matches
	Keywords map[string]int64 // ad hoc parameters, typically for experiments/tuning
	UsedKW   map[string]bool  // if a keyword is supplied to a phase, note that it was used.
}

var kwMu sync.Mutex

// DumpFileForPhase creates a file from the function name and phase name,
// warning and returning nil if this is not possible.
func (f *Func) DumpFileForPhase(phaseName string) io.WriteCloser {
	f.dumpFileSeq++
	fname := fmt.Sprintf("%s_%02d__%s.dump", f.Name, int(f.dumpFileSeq), phaseName)
	fname = strings.ReplaceAll(fname, " ", "_")
	fname = strings.ReplaceAll(fname, "/", "_")
	fname = strings.ReplaceAll(fname, ":", "_")

	if ssaDir := os.Getenv("GOSSADIR"); ssaDir != "" {
		fname = filepath.Join(ssaDir, fname)
	}

	fi, err := os.Create(fname)
	if err != nil {
		f.Warnl(src.NoXPos, "Unable to create after-phase dump file %s", fname)
		return nil
	}
	return fi
}

// DumpFile creates a file from the phase name and function name
// Dumping is done to files to avoid buffering huge strings before
// output.
func (f *Func) DumpFile(phaseName string) {
	fi := f.DumpFileForPhase(phaseName)
	if fi != nil {
		p := StringFuncPrinter{w: fi}
		FprintFunc(p, f)
		fi.Close()
	}
}

func (p *Pass) AddDump(s string) {
	if p.Dump == nil {
		p.Dump = make(map[string]bool)
	}
	p.Dump[s] = true
}

func (p *Pass) String() string {
	if p == nil {
		return "nil pass"
	}
	return p.Name
}

func (p *Pass) Val(kw string, ifUnset int64) int64 {
	if p == nil || p.Keywords == nil {
		return ifUnset
	}
	if v, ok := p.Keywords[kw]; ok {
		kwMu.Lock()
		p.UsedKW[kw] = true
		kwMu.Unlock()
		return v
	}
	return ifUnset
}
