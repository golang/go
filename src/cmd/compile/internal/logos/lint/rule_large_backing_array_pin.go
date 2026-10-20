// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"cmp"
	"fmt"
	"slices"

	"cmd/compile/internal/syntax"
)

// largeBackingArrayPin is perf::large_backing_array_pin (SPEC Sec 1.2,
// #158): a subslice or substring of a backing array larger than 64 KB
// escapes, so the whole array stays reachable for as long as the small part
// does. The compiler decides the escape inside escape analysis, a phase
// after Run: the rule has no Check, like a rule an earlier phase enforces,
// and RunPins resolves the findings the escape phase hands over. The rule is
// language-neutral: it reads the escape analysis of Go code and loGo/s code
// alike.
var largeBackingArrayPin = Rule{
	ID:    "perf::large_backing_array_pin",
	Group: Perf,
	Go:    true,
}

// PinFinding is one escaping subslice or substring of a large backing store,
// as escape analysis reports it: where the slice expression starts, how many
// bytes the backing store has, and whether the sliced value is a string.
type PinFinding struct {
	File      string
	Line, Col uint
	Bytes     int64
	Substring bool
}

// PinActive reports whether the escape phase must track large backing stores
// for in: the rule is on at the severity of the package and applies to its
// kind.
func PinActive(in *Input) bool {
	return ruleLevel(in, largeBackingArrayPin) != Allow
}

// RunPins resolves the findings of the escape phase for in like Run does for
// a rule: severity from the policy, the Go cap, and //logos:allow
// directives. It reports each source position once, because inlined bodies
// and generic instantiations are analyzed more than once, and it ignores a
// finding in a file the package does not hold: the body of a function from
// another package, inlined here, belongs to that package's own lint.
func RunPins(in *Input, found []PinFinding) []Result {
	r := largeBackingArrayPin
	r.Check = func(pass *Pass) {
		for _, f := range uniquePins(found) {
			if pos, ok := pinPos(in, f); ok {
				pass.Report(Diagnostic{Pos: pos, Message: f.message()})
			}
		}
	}
	return runRule(in, policyOf(in), collectDirectives(in), r)
}

// ruleLevel returns the severity at which rule r runs for in, or Allow when
// it does not apply to the kind of package.
func ruleLevel(in *Input, r Rule) Severity {
	if in.Kind == KindGo && !r.Go {
		return Allow
	}
	return policyOf(in).Level(r.ID, in.Kind)
}

// policyOf returns the policy of in, the default policy when it has none.
func policyOf(in *Input) *Policy {
	if in.Policy == nil {
		return DefaultPolicy()
	}
	return in.Policy
}

// uniquePins returns the findings ordered by position with duplicates of one
// position removed.
func uniquePins(found []PinFinding) []PinFinding {
	sorted := slices.Clone(found)
	slices.SortStableFunc(sorted, func(a, b PinFinding) int {
		return cmp.Or(cmp.Compare(a.File, b.File), cmp.Compare(a.Line, b.Line), cmp.Compare(a.Col, b.Col))
	})
	return slices.CompactFunc(sorted, func(a, b PinFinding) bool {
		return a.File == b.File && a.Line == b.Line && a.Col == b.Col
	})
}

// pinPos returns the syntax position of f, in the file base of the source
// file it names so that directives of that file apply. It reports false when
// the package holds no such file.
func pinPos(in *Input, f PinFinding) (syntax.Pos, bool) {
	return filePos(in, f.File, f.Line, f.Col)
}

// filePos returns the syntax position of line and col in the source file of
// in named file, in the file base of that file. It reports false when the
// package holds no such file.
func filePos(in *Input, file string, line, col uint) (syntax.Pos, bool) {
	for _, f := range in.Files {
		if f.Pos().RelFilename() == file {
			return syntax.MakePos(f.Pos().FileBase(), line, col), true
		}
	}
	return syntax.Pos{}, false
}

// message returns the diagnostic text of f.
func (f PinFinding) message() string {
	if f.Substring {
		return fmt.Sprintf("substring of a %d-byte backing string escapes and keeps the whole string reachable; copy the part you keep, for example with strings.Clone", f.Bytes)
	}
	return fmt.Sprintf("subslice of a %d-byte backing array escapes and keeps the whole array reachable; copy the part you keep, for example with slices.Clone", f.Bytes)
}
