// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"strconv"
	"strings"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// sprintfFile rewrites fmt.Sprintf calls with interpolation-compatible
// verbs to $"...". Calls inside recorded edit spans stay as written.
func (s *nextScan) sprintfFile(f *syntax.File) {
	syntax.Inspect(f, func(n syntax.Node) bool {
		call, ok := n.(*syntax.CallExpr)
		if !ok {
			return true
		}
		if s.insideLines(int(call.Pos().Line())) {
			return false
		}
		if rep, ok := s.sprintf(call); ok {
			start, ok := s.startOf(call)
			if !ok {
				return false
			}
			end, ok := s.endOf(call)
			if !ok {
				return false
			}
			trailing, blocked := scanComments(s.src, start, end, end)
			if blocked {
				return false
			}
			s.emit(start, end, rep+trailing)
			return false
		}
		return true
	})
}

// sprintf converts one Sprintf call to interpolation.
func (s *nextScan) sprintf(call *syntax.CallExpr) (string, bool) {
	if !s.sprintfCallee(call) {
		return "", false
	}
	cooked, ok := sprintfCooked(call)
	if !ok {
		return "", false
	}
	items, ok := parseVerbs(cooked)
	if !ok {
		return "", false
	}
	srcs, ok := s.sprintfOperands(items, call.ArgList[1:])
	if !ok {
		return "", false
	}
	return buildInterp(items, srcs), true
}

// sprintfCallee reports whether call targets fmt.Sprintf.
func (s *nextScan) sprintfCallee(call *syntax.CallExpr) bool {
	fun, ok := call.Fun.(*syntax.SelectorExpr)
	if !ok {
		return false
	}
	if fun.Sel.Value != "Sprintf" {
		return false
	}
	base, ok := fun.X.(*syntax.Name)
	if !ok {
		return false
	}
	pkg, ok := s.info.Uses[base].(*types2.PkgName)
	if !ok {
		return false
	}
	return pkg.Imported().Path() == "fmt"
}

// sprintfCooked unquotes the format literal of a Sprintf call.
func sprintfCooked(call *syntax.CallExpr) (string, bool) {
	if call.HasDots {
		return "", false
	}
	if len(call.ArgList) == 0 {
		return "", false
	}
	lit, ok := call.ArgList[0].(*syntax.BasicLit)
	if !ok {
		return "", false
	}
	if lit.Kind != syntax.StringLit {
		return "", false
	}
	if lit.Bad {
		return "", false
	}
	cooked, err := strconv.Unquote(lit.Value)
	if err != nil {
		return "", false
	}
	return cooked, true
}

// sprintfOperands pairs each verb chunk with its argument source text.
func (s *nextScan) sprintfOperands(items []verbItem, args []syntax.Expr) ([]string, bool) {
	if countVerbs(items) != len(args) {
		return nil, false
	}
	srcs := make([]string, 0, len(args))
	k := 0
	for _, it := range items {
		if it.verb == 0 {
			continue
		}
		src, ok := s.sprintfOperand(args[k], it.verb)
		if !ok {
			return nil, false
		}
		srcs = append(srcs, src)
		k++
	}
	return srcs, true
}

// sprintfOperand checks one verb argument against its verb and returns
// its source text.
func (s *nextScan) sprintfOperand(arg syntax.Expr, verb byte) (string, bool) {
	t := arg.GetTypeInfo().Type
	if t == nil {
		return "", false
	}
	kind, ok := types2.LogosInterpKindOf(t)
	if !ok {
		return "", false
	}
	if !verbKind(verb, kind) {
		return "", false
	}
	end, ok := s.endOf(arg)
	if !ok {
		return "", false
	}
	start, ok := s.startOf(arg)
	if !ok {
		return "", false
	}
	return string(s.src[start:end]), true
}

// verbItem is one format chunk: literal text plus an optional verb.
type verbItem struct {
	lit  string
	verb byte
}

// parseVerbs splits a format string into literal and verb chunks. Only
// s, d, and v verbs qualify; anything else rejects the call.
func parseVerbs(cooked string) ([]verbItem, bool) {
	var items []verbItem
	var lit strings.Builder
	flush := func(verb byte) {
		items = append(items, verbItem{lit: lit.String(), verb: verb})
		lit.Reset()
	}
	i := 0
	for i < len(cooked) {
		if cooked[i] != '%' {
			lit.WriteByte(cooked[i])
			i++
			continue
		}
		i++
		if i >= len(cooked) {
			return nil, false
		}
		c := cooked[i]
		i++
		if c == '%' {
			lit.WriteByte('%')
			continue
		}
		if c == 's' || c == 'd' || c == 'v' {
			flush(c)
			continue
		}
		return nil, false
	}
	flush(0)
	return items, true
}

// countVerbs counts the verb chunks of parsed items.
func countVerbs(items []verbItem) int {
	n := 0
	for _, it := range items {
		if it.verb != 0 {
			n++
		}
	}
	return n
}

// verbKind reports whether a verb renders a kind identically under
// fmt and interpolation.
func verbKind(verb byte, kind types2.LogosInterpKind) bool {
	switch verb {
	case 's':
		return verbKindS(kind)
	case 'd':
		return verbKindD(kind)
	case 'v':
		return verbKindV(kind)
	}
	return false
}

// verbKindS reports whether a kind renders identically under %s and
// interpolation.
func verbKindS(kind types2.LogosInterpKind) bool {
	return kind == types2.LogosInterpString || kind == types2.LogosInterpBytes || kind == types2.LogosInterpError || kind == types2.LogosInterpStringer
}

// verbKindD reports whether a kind renders identically under %d and
// interpolation.
func verbKindD(kind types2.LogosInterpKind) bool {
	return kind == types2.LogosInterpInt || kind == types2.LogosInterpUint
}

// verbKindV reports whether a kind renders identically under %v and
// interpolation.
func verbKindV(kind types2.LogosInterpKind) bool {
	return kind == types2.LogosInterpString || kind == types2.LogosInterpInt || kind == types2.LogosInterpUint || kind == types2.LogosInterpBool || kind == types2.LogosInterpError || kind == types2.LogosInterpStringer
}

// buildInterp assembles the interpolated string from chunks and
// argument sources.
func buildInterp(items []verbItem, srcs []string) string {
	var out strings.Builder
	out.WriteString(`$"`)
	k := 0
	for _, it := range items {
		out.WriteString(quoteChunk(it.lit))
		if it.verb != 0 {
			out.WriteByte('{')
			out.WriteString(srcs[k])
			out.WriteByte('}')
			k++
		}
	}
	out.WriteByte('"')
	return out.String()
}

// quoteChunk escapes one literal chunk for an interpreted string:
// braces double, then standard quoting without its quotes.
func quoteChunk(chunk string) string {
	doubled := strings.ReplaceAll(strings.ReplaceAll(chunk, "{", "{{"), "}", "}}")
	quoted := strconv.Quote(doubled)
	return quoted[1 : len(quoted)-1]
}
