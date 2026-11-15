// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"fmt"
	"slices"
	"strings"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// linknameMismatch is correctness::linkname_mismatch (SPEC Sec 1.2): a
// //go:linkname local pkg.name whose pkg is a direct import of this package,
// whose symbol is visible in that import's export data, and whose signature
// differs from the local declaration. A remote the import graph cannot
// resolve is not checked, because there is no export data to compare.
// Language neutral.
var linknameMismatch = Rule{
	ID:    "correctness::linkname_mismatch",
	Group: Correctness,
	Go:    true,
	Check: checkLinknameMismatch,
}

func checkLinknameMismatch(pass *Pass) {
	locals := packageObjects(pass)
	for _, f := range pass.Files {
		src, ok := pass.in.Sources[f]
		if !ok || !strings.Contains(string(src), "go:linkname ") {
			continue
		}
		for _, ln := range linknameComments(f, src) {
			reportLinknameMismatch(pass, locals, ln)
		}
	}
}

// linknameComment is one //go:linkname directive.
type linknameComment struct {
	Local  string
	Remote string
	Pos    syntax.Pos
}

func linknameComments(f *syntax.File, src []byte) []linknameComment {
	var out []linknameComment
	lines := strings.Split(string(src), "\n")
	base := f.Pos().FileBase()
	mode := syntax.Mode(0)
	syntax.CommentsDoMode(strings.NewReader(string(src)), mode, func(line, col uint, text string) {
		text = strings.TrimRight(text, " \t\r")
		if !strings.HasPrefix(text, "//go:linkname ") {
			return
		}
		fields := strings.Fields(strings.TrimPrefix(text, "//go:linkname "))
		if len(fields) != 2 {
			return
		}
		if line == 0 || int(line) > len(lines) {
			return
		}
		out = append(out, linknameComment{
			Local:  fields[0],
			Remote: fields[1],
			Pos:    syntax.MakePos(base, line, col),
		})
	})
	return out
}

// packageObjects indexes the package-level functions and variables by name.
func packageObjects(pass *Pass) map[string]types2.Object {
	out := make(map[string]types2.Object)
	if pass.in.Pkg == nil {
		return out
	}
	scope := pass.in.Pkg.Scope()
	for _, name := range scope.Names() {
		out[name] = scope.Lookup(name)
	}
	return out
}

func reportLinknameMismatch(pass *Pass, locals map[string]types2.Object, ln linknameComment) {
	local := locals[ln.Local]
	if local == nil {
		return
	}
	pos := local.Pos()
	if !pos.IsKnown() {
		pos = ln.Pos
	}
	pkgPath, remote, ok := strings.Cut(ln.Remote, ".")
	if !ok {
		return
	}
	imported := packageInImportGraph(pass, pkgPath)
	if imported == nil {
		return
	}
	target := linkedSymbol(imported, remote)
	if target == nil {
		return
	}
	localType := local.Type()
	remoteType := methodExpressionType(target.Type())
	if types2.Identical(localType, remoteType) {
		return
	}
	pass.Report(Diagnostic{
		Pos: pos,
		End: pos,
		Message: fmt.Sprintf("linkname %s has signature %s; %s declares %s",
			ln.Remote, types2.TypeString(remoteType, nil), ln.Local, types2.TypeString(localType, nil)),
	})
}

func methodExpressionType(t types2.Type) types2.Type {
	sig, ok := t.(*types2.Signature)
	if !ok || sig.Recv() == nil {
		return t
	}
	vars := make([]*types2.Var, 0, sig.Params().Len()+1)
	vars = append(vars, sig.Recv())
	for i := range sig.Params().Len() {
		vars = append(vars, sig.Params().At(i))
	}
	return types2.NewSignatureType(nil, nil, nil, types2.NewTuple(vars...), sig.Results(), sig.Variadic())
}

func packageInImportGraph(pass *Pass, path string) *types2.Package {
	if pass.in.Pkg == nil {
		return nil
	}
	seen := make(map[*types2.Package]bool)
	queue := slices.Clone(pass.in.Pkg.Imports())
	for len(queue) > 0 {
		imp := queue[0]
		queue = queue[1:]
		if seen[imp] {
			continue
		}
		seen[imp] = true
		if imp.Path() == path {
			return imp
		}
		queue = append(queue, imp.Imports()...)
	}
	return nil
}

// linkedSymbol resolves name in pkg: a plain function or variable, or a
// method spelled Type.Method or (*Type).Method.
func linkedSymbol(pkg *types2.Package, name string) types2.Object {
	if obj := pkg.Scope().Lookup(name); obj != nil {
		return obj
	}
	recv, method, ok := splitMethodName(name)
	if !ok {
		return nil
	}
	obj := pkg.Scope().Lookup(recv)
	if obj == nil {
		return nil
	}
	named, isNamed := types2.Unalias(obj.Type()).(*types2.Named)
	if !isNamed {
		return nil
	}
	if p, isPtr := types2.Unalias(named).(*types2.Pointer); isPtr {
		named, _ = types2.Unalias(p.Elem()).(*types2.Named)
	}
	if named == nil {
		return nil
	}
	for i := range named.NumMethods() {
		if m := named.Method(i); m.Name() == method {
			return m
		}
	}
	return nil
}

// splitMethodName splits Type.Method and (*Type).Method spellings.
func splitMethodName(name string) (string, string, bool) {
	if strings.HasPrefix(name, "(*") {
		rest, ok := strings.CutPrefix(name, "(*")
		if !ok {
			return "", "", false
		}
		recv, method, ok := strings.Cut(rest, ").")
		return recv, method, ok
	}
	recv, method, ok := strings.Cut(name, ".")
	return recv, method, ok
}
