// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// LowerGoCalls wraps arguments of statically-called loGo/s functions
// and methods with non-nil pointer parameters in MustNonNil for Go
// packages importing loGo/s. It reports whether it changed anything.
// Calls through func values, method values, and interfaces are a
// documented hole and stay unwrapped. Ejected output keeps no checks.
func LowerGoCalls(pkg *types2.Package, info *types2.Info, files []*syntax.File) bool {
	if pkg == nil || info == nil || !isUserModule(pkg.Path()) {
		return false
	}
	changed := false
	seq := 0
	for _, file := range files {
		if file == nil {
			continue
		}
		g := &goCallLower{pkg: pkg, info: info, file: file, siteSeq: seq}
		g.baseName = filepath.Base(file.Pos().FileBase().Filename())
		if g.run() {
			changed = true
		}
		seq = g.siteSeq
	}
	return changed
}

// isUserModule reports whether path belongs to a user module: the first
// path element contains a dot (example.com/...). Toolchain packages
// (std, cmd/..., logos/...) and command-line-arguments stay out, so the
// hook never wraps the compiler itself.
func isUserModule(path string) bool {
	if path == "command-line-arguments" || path == "main" {
		return true
	}
	first, _, _ := strings.Cut(path, "/")
	return strings.Contains(first, ".")
}

type goCallLower struct {
	pkg      *types2.Package
	info     *types2.Info
	file     *syntax.File
	baseName string
	siteSeq  int
	sites    []*syntax.VarDecl
	needRT   bool
}

func (g *goCallLower) run() bool {
	changed := false
	syntax.Inspect(g.file, func(n syntax.Node) bool {
		call, ok := n.(*syntax.CallExpr)
		if !ok {
			return true
		}
		if g.wrapStaticCall(call) {
			changed = true
		}
		return true
	})
	if g.needRT {
		g.file.DeclList = append([]syntax.Decl{rtImport(g.file.Pos())}, g.file.DeclList...)
		for _, d := range g.sites {
			g.file.DeclList = append(g.file.DeclList, d)
		}
	}
	return changed
}

// wrapStaticCall wraps non-nil pointer args of a static loGo/s callee.
func (g *goCallLower) wrapStaticCall(call *syntax.CallExpr) bool {
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok || isMustNonNilCall(call.Fun) {
		return false
	}
	if s := g.info.Selections[sel]; s != nil {
		return g.wrapMethodCall(call, sel, s)
	}
	fn, ok := g.info.Uses[sel.Sel].(*types2.Func)
	if !ok || fn == nil || !hasLogosFacts(fn.Pkg()) {
		return false
	}
	sig, ok := fn.Type().(*types2.Signature)
	if !ok || sig.Recv() != nil {
		return false
	}
	return g.wrapArgs(call, sig, sel.Pos(), fn)
}

// wrapMethodCall wraps a direct method call on a concrete receiver.
func (g *goCallLower) wrapMethodCall(call *syntax.CallExpr, sel *syntax.SelectorExpr, s *types2.Selection) bool {
	obj, ok := s.Obj().(*types2.Func)
	if !ok || obj == nil || !hasLogosFacts(obj.Pkg()) {
		return false
	}
	if recv := s.Recv(); recv != nil {
		if _, ok := types2.Unalias(recv).Underlying().(*types2.Interface); ok {
			return false
		}
	}
	sig, ok := obj.Type().(*types2.Signature)
	if !ok {
		return false
	}
	return g.wrapArgs(call, sig, sel.Pos(), obj)
}

// wrapArgs wraps each arg for a non-nil pointer param without opt cover.
func (g *goCallLower) wrapArgs(call *syntax.CallExpr, sig *types2.Signature, pos syntax.Pos, fn *types2.Func) bool {
	params := sig.Params()
	if params == nil {
		return false
	}
	covered := optParamSet(fn)
	changed := false
	for i := range params.Len() {
		if i >= len(call.ArgList) || (sig.Variadic() && i >= params.Len()-1) {
			break
		}
		if covered[i] {
			continue
		}
		if _, ok := types2.Unalias(params.At(i).Type()).(*types2.Pointer); !ok {
			continue
		}
		arg := call.ArgList[i]
		if arg == nil || isMustNonNilCall(arg) {
			continue
		}
		site := g.newSite(arg.Pos())
		fun := syntax.Expr(rtSel(arg.Pos(), "MustNonNil"))
		if isNilLiteral(arg) {
			if targ, ok := explicitTypeArg(params.At(i).Type(), arg.Pos()); ok {
				fun = at(&syntax.IndexExpr{X: fun, Index: targ}, arg.Pos())
			}
		}
		call.ArgList[i] = callOf(arg.Pos(), fun, []syntax.Expr{
			arg,
			at(&syntax.Operation{Op: syntax.And, X: nameOf(arg.Pos(), site)}, arg.Pos()),
		})
		changed = true
	}
	return changed
}

func (g *goCallLower) newSite(pos syntax.Pos) string {
	name := fmt.Sprintf("_logos_site%d", g.siteSeq)
	g.siteSeq++
	base, line := siteDisplay(pos, g.baseName, pos.Line())
	g.sites = append(g.sites, siteDecl(pos, nameOf(pos, name), base, line))
	g.needRT = true
	return name
}

// isNilLiteral reports whether e is the predeclared nil identifier.
func isNilLiteral(e syntax.Expr) bool {
	name, ok := syntax.Unparen(e).(*syntax.Name)
	return ok && name.Value == "nil"
}

// explicitTypeArg spells a pointer param type for MustNonNil[T] when the
// argument is an untyped nil literal and T cannot be inferred. It
// handles pointer to named (qualified by its package name) and pointer
// to basic kinds; other shapes rely on inference from typed arguments.
func explicitTypeArg(pt types2.Type, pos syntax.Pos) (syntax.Expr, bool) {
	ptr, ok := types2.Unalias(pt).(*types2.Pointer)
	if !ok {
		return nil, false
	}
	base := types2.Unalias(ptr.Elem())
	if b, ok := base.(*types2.Basic); ok {
		return at(&syntax.Operation{Op: syntax.Mul, X: nameOf(pos, b.Name())}, pos), true
	}
	if n, ok := base.(*types2.Named); ok && n.Obj() != nil {
		pkg := ""
		if qp := n.Obj().Pkg(); qp != nil {
			pkg = qp.Name()
		}
		var elem syntax.Expr = nameOf(pos, n.Obj().Name())
		if pkg != "" {
			elem = selOf(pos, nameOf(pos, pkg), n.Obj().Name())
		}
		return at(&syntax.Operation{Op: syntax.Mul, X: elem}, pos), true
	}
	return nil, false
}

func isMustNonNilCall(e syntax.Expr) bool {
	call, ok := syntax.Unparen(e).(*syntax.CallExpr)
	if !ok {
		return false
	}
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok {
		return false
	}
	x, ok := syntax.Unparen(sel.X).(*syntax.Name)
	return ok && x.Value == rtAlias && sel.Sel != nil && sel.Sel.Value == "MustNonNil"
}

func hasLogosFacts(pkg *types2.Package) bool {
	if pkg == nil {
		return false
	}
	f, ok := facts.Lookup(pkg.Path())
	return ok && f != nil && !f.GoPackage
}

// optParamSet reports which param indices carry opt facts.
func optParamSet(fn *types2.Func) map[int]bool {
	out := map[int]bool{}
	if fn == nil || fn.Pkg() == nil {
		return out
	}
	f, ok := facts.Lookup(fn.Pkg().Path())
	if !ok || f == nil {
		return out
	}
	obj := fn.Name()
	if sig, ok := fn.Type().(*types2.Signature); ok && sig.Recv() != nil {
		recv := sig.Recv().Type()
		if _, ok := types2.Unalias(recv).(*types2.Pointer); ok {
			if p, ok := types2.Unalias(recv).(*types2.Pointer); ok {
				if n, ok := types2.Unalias(p.Elem()).(*types2.Named); ok && n.Obj() != nil {
					obj = "(*" + n.Obj().Name() + ")." + fn.Name()
				}
			}
		} else if n, ok := types2.Unalias(recv).(*types2.Named); ok && n.Obj() != nil {
			obj = n.Obj().Name() + "." + fn.Name()
		}
	}
	for _, o := range f.Opts {
		if o.Object != obj {
			continue
		}
		head, _, _ := strings.Cut(o.Path, ".")
		if len(head) < 2 || head[0] != 'P' {
			continue
		}
		if idx, err := strconv.Atoi(head[1:]); err == nil {
			out[idx] = true
		}
	}
	return out
}
