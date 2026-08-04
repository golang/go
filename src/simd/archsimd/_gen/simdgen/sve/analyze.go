// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sve

import (
	"fmt"
	"sort"
	"strings"

	"simd/archsimd/_gen/unify"
)

// classify decides what to do with an instruction and why. It returns the
// emitted defs (nil if skipped), a human-readable reason, and whether the
// reason is an anomaly, i.e. a case the loader does not understand and that
// should be investigated. Skips that are understood but deferred, e.g. memory,
// immediate, register list, special or non-uniform operands are not anomalies.
//
// emitAll and analyze share this logic so that what gets emitted and what gets
// reported can never drift apart.
func (inst *Instruction) classify() (defs []*unify.Value, reason string, anomaly bool) {
	if !inst.isSVE() {
		return nil, "not an SVE instruction", false
	}
	if inst.isAlias() {
		return nil, "alias", false
	}
	allEncOps := inst.allEncodingOperands()
	if len(allEncOps) == 0 {
		// Nullary or unsized forms (e.g. SETFFR): nothing to emit, but not a
		// parse failure.
		return nil, "no operands (deferred)", false
	}

	// emit every distinct encoding form. An anomaly in any form is reported for
	// the whole instruction; a form that is merely deferred contributes no defs
	// but does not fail the others.
	var skip string
	for _, ops := range allEncOps {
		d, r, a := inst.classifyOperands(ops)
		if a {
			return nil, r, true
		}
		if len(d) == 0 {
			if skip == "" {
				skip = r
			}
			continue
		}
		defs = append(defs, d...)
	}
	if len(defs) == 0 {
		return nil, skip, false
	}
	return defs, "", false
}

// classifyOperands is classify for a single encoding form's operands.
func (inst *Instruction) classifyOperands(ops []Operand) (defs []*unify.Value, reason string, anomaly bool) {
	// Unrecognized operands are anomalies.
	for _, op := range ops {
		if op.Class == "unknown" {
			return nil, fmt.Sprintf("unknown operand %q", op.Raw), true
		}
	}
	// Register lists are not modeled yet.
	// TODO: emit list operands, which needs regalloc support for register lists,
	// instead of skipping the instruction.
	if hasClass(ops, "reglist") {
		return nil, "register list (deferred, TODO)", false
	}

	// Every arrangement symbol used by an operand must resolve to a real size
	// table; if not, the loader does not understand the instruction.
	for _, link := range arngLinks(ops) {
		if len(inst.resolveArrangementTable(link)) == 0 {
			return nil, fmt.Sprintf("arrangement %q resolves to empty domain", link), true
		}
	}

	defs = inst.emitVariants(ops)
	if len(defs) == 0 {
		return nil, "no defs emitted (all rows reserved/filtered)", false
	}
	return defs, "", false
}

// report is the outcome of analyzing a corpus of SVE instructions.
type report struct {
	Total   int            // SVE instructions considered
	Emitted int            // instructions that produced at least one def
	Defs    int            // total defs emitted
	Reasons map[string]int // skip/emit reason -> instruction count
	// Anomalies lists "<mnemonic> (<file title>): <reason>" for every
	// instruction the loader did not understand.
	Anomalies []string
}

// analyze parses the ARM64 ISA XML files at path and reports, for every SVE
// instruction, whether it was emitted or skipped and why, collecting the
// unrecognized cases in report.Anomalies. Used by the corpus test.
func analyze(path string) (*report, error) {
	insts, err := parseInstructions(path)
	if err != nil {
		return nil, err
	}
	r := &report{Reasons: map[string]int{}}
	for _, inst := range insts {
		r.Total++
		defs, reason, anomaly := inst.classify()
		key := reason
		if key == "" {
			key = "emitted"
			r.Emitted++
			r.Defs += len(defs)
		}
		r.Reasons[key]++
		if anomaly {
			r.Anomalies = append(r.Anomalies,
				fmt.Sprintf("%s (%s): %s", inst.mnemonic(), inst.Title, reason))
		}
	}
	sort.Strings(r.Anomalies)
	return r, nil
}

// String renders a human-readable summary of the report.
func (r *report) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "SVE instructions: %d (%d emitted -> %d defs)\n", r.Total, r.Emitted, r.Defs)
	fmt.Fprintf(&b, "disposition:\n")
	keys := make([]string, 0, len(r.Reasons))
	for k := range r.Reasons {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if r.Reasons[keys[i]] != r.Reasons[keys[j]] {
			return r.Reasons[keys[i]] > r.Reasons[keys[j]]
		}
		return keys[i] < keys[j]
	})
	for _, k := range keys {
		fmt.Fprintf(&b, "  %5d  %s\n", r.Reasons[k], k)
	}
	fmt.Fprintf(&b, "anomalies: %d\n", len(r.Anomalies))
	for _, a := range r.Anomalies {
		fmt.Fprintf(&b, "  ! %s\n", a)
	}
	return b.String()
}
