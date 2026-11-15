// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"fmt"
	"go/constant"

	"cmd/compile/internal/types2"
	"internal/buildcfg"
)

// layoutConstant is correctness::layout_constant_mismatch (SPEC Sec 1.2): a
// constant marked //logos:sizeof, //logos:offsetof, or //logos:alignof must
// equal the compiler's value for the named type or field. The diagnostic
// states the declared value and the actual one. Language neutral.
var layoutConstant = Rule{
	ID:    layoutRule,
	Group: Correctness,
	Go:    true,
	Check: checkLayoutConstants,
}

func checkLayoutConstants(pass *Pass) {
	dirs := scanDirectives(pass.in)
	for _, r := range dirs.Misplaced {
		if r.Rule == layoutRule {
			pass.Report(diagOf(r))
		}
	}
	sizes := types2.SizesFor("gc", buildcfg.GOARCH)
	for _, l := range dirs.Layouts {
		actual, ok := layoutValue(sizes, pass, l)
		if !ok {
			pass.Report(Diagnostic{
				Pos:     l.Name.Pos(),
				End:     l.Name.Pos(),
				Message: fmt.Sprintf("%s %s: the type or field does not resolve", l.Kind, l.Target),
			})
			continue
		}
		declared, ok := constInt64(l.Name.Val())
		if !ok {
			pass.Report(Diagnostic{
				Pos:     l.Name.Pos(),
				End:     l.Name.Pos(),
				Message: fmt.Sprintf("%s declares a non-integer constant", l.Name.Name()),
			})
			continue
		}
		if declared != actual {
			pass.Report(Diagnostic{
				Pos:     l.Name.Pos(),
				End:     l.Name.Pos(),
				Message: fmt.Sprintf("%s declares %d, %s %s is %d", l.Name.Name(), declared, l.Kind, l.Target, actual),
			})
		}
	}
}

// layoutValue returns the compiler's value for one layout directive.
func layoutValue(sizes types2.Sizes, pass *Pass, l layoutCheck) (int64, bool) {
	named, fieldPath, ok := resolveLayoutTarget(pass, l.Target)
	if !ok {
		return 0, false
	}
	switch l.Kind {
	case "sizeof":
		return sizes.Sizeof(named), true
	case "alignof":
		return sizes.Alignof(named), true
	case "offsetof":
		return offsetOf(sizes, named, fieldPath)
	}
	return 0, false
}

// resolveLayoutTarget resolves T, T.f, or T.f.g in the package scope.
func resolveLayoutTarget(pass *Pass, target string) (types2.Type, []string, bool) {
	parts := splitDotted(target)
	if len(parts) == 0 || pass.in.Pkg == nil {
		return nil, nil, false
	}
	obj := pass.in.Pkg.Scope().Lookup(parts[0])
	if obj == nil {
		obj = types2.Universe.Lookup(parts[0])
	}
	if obj == nil {
		return nil, nil, false
	}
	t := obj.Type()
	if len(parts) == 1 {
		return t, nil, true
	}
	return t, parts[1:], true
}

// offsetOf returns the byte offset of the named field path inside t.
func offsetOf(sizes types2.Sizes, t types2.Type, fields []string) (int64, bool) {
	offset := int64(0)
	cur := t
	for _, name := range fields {
		st, ok := structOf(cur)
		if !ok {
			return 0, false
		}
		idx := fieldIndex(st, name)
		if idx < 0 {
			return 0, false
		}
		offset += sizes.Offsetsof(structFields(st))[idx]
		cur = st.Field(idx).Type()
	}
	return offset, true
}

// structFields returns the fields of st as types2 variables for Offsetsof.
func structFields(st *types2.Struct) []*types2.Var {
	out := make([]*types2.Var, st.NumFields())
	for i := range out {
		out[i] = st.Field(i)
	}
	return out
}

func fieldIndex(st *types2.Struct, name string) int {
	for i := range st.NumFields() {
		if st.Field(i).Name() == name {
			return i
		}
	}
	return -1
}

// constInt64 returns an integer constant value.
func constInt64(v any) (val int64, ok bool) {
	if cv, isConst := v.(constant.Value); isConst {
		return constant.Int64Val(cv)
	}
	switch v := v.(type) {
	case int64:
		return v, true
	case int:
		return int64(v), true
	case uint64:
		if v > 1<<63-1 {
			return 0, false
		}
		return int64(v), true
	}
	return 0, false
}

func splitDotted(s string) []string {
	var out []string
	start := 0
	for i := 0; i <= len(s); i++ {
		if i == len(s) || s[i] == '.' {
			if i > start {
				out = append(out, s[start:i])
			}
			start = i + 1
		}
	}
	return out
}
