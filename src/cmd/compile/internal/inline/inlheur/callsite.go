// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/internal/src"
	"fmt"
	"io"
	"path/filepath"
	"sort"
	"strings"
)

// CallSite records useful information about a potentially inlinable
// (direct) function call. "Callee" is the target of the call, "Call"
// is the ir node corresponding to the call itself, "Assign" is
// the top-level assignment statement containing the call (if the call
// appears in the form of a top-level statement, e.g. "x := foo()"),
// "Flags" contains properties of the call that might be useful for
// making inlining decisions, "Score" is the final score assigned to
// the site, and "ID" is a numeric ID for the site within its
// containing function.
type CallSite struct {
	Callee *ir.Func
	Call   *ir.CallExpr
	parent *CallSite
	Assign ir.Node
	Flags  CSPropBits

	ArgProps  []ActualExprPropBits
	Score     int
	ScoreMask scoreAdjustTyp
	ID        uint
	aux       uint8
}

// CallSiteTab is a table of call sites, keyed by call expr.
// Ideally it would be nice to key the table by src.XPos, but
// this results in collisions for calls on very long lines (the
// front end saturates column numbers at 255). We also wind up
// with many calls that share the same auto-generated pos.
type CallSiteTab map[*ir.CallExpr]*CallSite

// ActualExprPropBits describes a property of an actual expression (value
// passed to some specific func argument at a call site).
type ActualExprPropBits uint8

const (
	ActualExprConstant ActualExprPropBits = 1 << iota
	ActualExprIsConcreteConvIface
	ActualExprIsFunc
	ActualExprIsInlinableFunc
)

type CSPropBits uint32

const (
	CallSiteInLoop CSPropBits = 1 << iota
	CallSiteOnPanicPath
	CallSiteInInitFunc
)

type csAuxBits uint8

const (
	csAuxInlined = 1 << iota
)

// encodedCallSiteTab is a table keyed by "encoded" callsite
// (stringified src.XPos plus call site ID) mapping to a value of call
// property bits and score.
type encodedCallSiteTab map[string]propsAndScore

type propsAndScore struct {
	props CSPropBits
	score int
	mask  scoreAdjustTyp
}

func (pas propsAndScore) String() string {
	return fmt.Sprintf("P=%s|S=%d|M=%s", pas.props.String(),
		pas.score, pas.mask.String())
}

func (cst CallSiteTab) merge(other CallSiteTab) error {
	for k, v := range other {
		if prev, ok := cst[k]; ok {
			return fmt.Errorf("internal error: collision during call site table merge, fn=%s callsite=%s", prev.Callee.Sym().Name, fmtFullPos(prev.Call.Pos()))
		}
		cst[k] = v
	}
	return nil
}

func fmtFullPos(p src.XPos) string {
	var sb strings.Builder
	sep := ""
	base.Ctxt.AllPos(p, func { pos ->
		fmt.Fprintf(&sb, sep)
		sep = "|"
		file := filepath.Base(pos.Filename())
		fmt.Fprintf(&sb, "%s:%d:%d", file, pos.Line(), pos.Col())
	})
	return sb.String()
}

func EncodeCallSiteKey(cs *CallSite) string {
	var sb strings.Builder
	// FIXME: maybe rewrite line offsets relative to function start?
	sb.WriteString(fmtFullPos(cs.Call.Pos()))
	fmt.Fprintf(&sb, "|%d", cs.ID)
	return sb.String()
}

func buildEncodedCallSiteTab(tab CallSiteTab) encodedCallSiteTab {
	r := make(encodedCallSiteTab)
	for _, cs := range tab {
		k := EncodeCallSiteKey(cs)
		r[k] = propsAndScore{
			props: cs.Flags,
			score: cs.Score,
			mask:  cs.ScoreMask,
		}
	}
	return r
}

// dumpCallSiteComments emits comments into the dump file for the
// callsites in the function of interest. If "ecst" is non-nil, we use
// that, otherwise generated a fresh encodedCallSiteTab from "tab".
func dumpCallSiteComments(w io.Writer, tab CallSiteTab, ecst encodedCallSiteTab) {
	if ecst == nil {
		ecst = buildEncodedCallSiteTab(tab)
	}
	tags := make([]string, 0, len(ecst))
	for k := range ecst {
		tags = append(tags, k)
	}
	sort.Strings(tags)
	for _, s := range tags {
		v := ecst[s]
		fmt.Fprintf(w, "// callsite: %s flagstr %q flagval %d score %d mask %d maskstr %q\n", s, v.props.String(), v.props, v.score, v.mask, v.mask.String())
	}
	fmt.Fprintf(w, "// %s\n", csDelimiter)
}
