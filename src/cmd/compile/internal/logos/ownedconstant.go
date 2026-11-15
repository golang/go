// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/logos/ctfe"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmd/internal/logosowned"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"go/ast"
	"go/build/constraint"
	"go/parser"
	"go/token"
	"internal/logosfile"
	"maps"
	"slices"
	"strconv"
	"strings"
)

type ownedPending struct {
	owner  string
	pkg    string
	source string
	decls  []logosowned.Declaration
}

var ownedPendingList []ownedPending
var ownedPendingSeen map[string]bool
var ownedOwnerImports map[string]map[string]string

func ownedOwnerImport(owner, dependency string) string {
	if ownedOwnerImports == nil {
		ownedOwnerImports = make(map[string]map[string]string)
	}
	imports := ownedOwnerImports[owner]
	if imports == nil {
		imports = make(map[string]string)
		ownedOwnerImports[owner] = imports
	}
	if qualifier, ok := imports[dependency]; ok {
		return qualifier
	}
	sum := sha256.Sum256([]byte(dependency))
	qualifier := "_logos_owner_" + hex.EncodeToString(sum[:6])
	imports[dependency] = qualifier
	return qualifier
}

var ownedTypeNodes map[syntax.Expr]types2.LogosOwnedType
var ownedWitnessByType map[types2.Type]types2.LogosOwnedType

func clearOwnedConstants() {
	ownedOwnerImports = nil
	ownedPendingList = nil
	ownedPendingSeen = nil
	ownedTypeNodes = nil
	ownedFileSeq = nil
	ownedEjectImports = nil
	ownedWitnessByType = nil
}

func OwnedConstantRequests() ([]logosowned.Request, error) {
	out := make([]logosowned.Request, 0, len(ownedPendingList))
	seen := make(map[string]bool)
	for _, p := range ownedPendingList {
		req, err := logosowned.NewRequest(p.owner, p.pkg, p.source, p.decls)
		if err != nil {
			return nil, err
		}
		key := req.Owner + "\x00" + req.ID + "\x00" + ownedDeclsKey(req.Declarations)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, req)
	}
	slices.SortFunc(out, func(a, b logosowned.Request) int {
		if c := strings.Compare(a.Owner, b.Owner); c != 0 {
			return c
		}
		if c := strings.Compare(a.Package, b.Package); c != 0 {
			return c
		}
		if c := strings.Compare(a.ID, b.ID); c != 0 {
			return c
		}
		return strings.Compare(ownedDeclsKey(a.Declarations), ownedDeclsKey(b.Declarations))
	})
	return out, nil
}

func OwnedTypeNodes() map[syntax.Expr]types2.LogosOwnedType {
	return maps.Clone(ownedTypeNodes)
}

func RegisterOwnedNode(n syntax.Expr, d types2.LogosOwnedType) {
	if n == nil {
		return
	}
	if ownedTypeNodes == nil {
		ownedTypeNodes = make(map[syntax.Expr]types2.LogosOwnedType)
	}
	ownedTypeNodes[n] = d
}

func ownedAliasFor(factoryPath, factoryName string, path []types2.LogosOwnedStep, kind string) string {
	h := sha256.New()
	h.Write([]byte(factoryPath))
	h.Write([]byte{0})
	h.Write([]byte(factoryName))
	h.Write([]byte{0})
	for _, s := range path {
		h.Write([]byte{byte(s.Kind)})
		var buf [8]byte
		v := s.Index
		for i := 7; i >= 0; i-- {
			buf[i] = byte(v)
			v >>= 8
		}
		h.Write(buf[:])
		h.Write([]byte{0})
	}
	h.Write([]byte(kind))
	sum := h.Sum(nil)
	return "LogosOwned_" + hex.EncodeToString(sum[:])
}

func ownedValueAliasFor(factoryPath, factoryName string, path []types2.LogosOwnedStep, kind, value string) string {
	h := sha256.New()
	h.Write([]byte(factoryPath))
	h.Write([]byte{0})
	h.Write([]byte(factoryName))
	h.Write([]byte{0})
	for _, s := range path {
		h.Write([]byte{byte(s.Kind)})
		var buf [8]byte
		v := s.Index
		for i := 7; i >= 0; i-- {
			buf[i] = byte(v)
			v >>= 8
		}
		h.Write(buf[:])
		h.Write([]byte{0})
	}
	h.Write([]byte(kind))
	h.Write([]byte{0})
	h.Write([]byte(value))
	sum := h.Sum(nil)
	return "LogosOwned_" + hex.EncodeToString(sum[:])
}

var ownedFileSeq map[*syntax.File]int

func ownedNextSeq(f *syntax.File) int {
	if ownedFileSeq == nil {
		ownedFileSeq = make(map[*syntax.File]int)
	}
	n := ownedFileSeq[f]
	ownedFileSeq[f] = n + 1
	return n
}

func ownedNativePlaceholder(fl *fileLower, role string) string {
	return fmt.Sprintf("_logos_owned_t%d_%s", ownedNextSeq(fl.file), role)
}

func (fl *fileLower) emitOwnedRequest(owner, pkg, source string, decls []logosowned.Declaration) bool {
	if base.LogosEject == "" {
		return true
	}
	if imports := ownedOwnerImports[owner]; len(imports) > 0 {
		packageClause, rest, ok := strings.Cut(source, "\n")
		if !ok {
			return false
		}
		paths := slices.Sorted(maps.Keys(imports))
		var header strings.Builder
		header.WriteString(packageClause)
		header.WriteByte('\n')
		used := map[string]bool{}
		if f, err := parser.ParseFile(token.NewFileSet(), "owned.go", []byte(source), parser.SkipObjectResolution); err == nil {
			ast.Inspect(f, func(n ast.Node) bool {
				if sel, ok := n.(*ast.SelectorExpr); ok {
					if id, ok := sel.X.(*ast.Ident); ok {
						used[id.Name] = true
					}
				}
				return true
			})
		}
		for _, path := range paths {
			if used[imports[path]] {
				fmt.Fprintf(&header, "import %s %q\n", imports[path], path)
			}
		}
		source = header.String() + rest
	}
	source = ownedConstraintHeader(fl) + source
	if ownedPendingSeen == nil {
		ownedPendingSeen = make(map[string]bool)
	}
	key := owner + "\x00" + pkg + "\x00" + source + "\x00" + ownedPendingDeclsKey(decls)
	if ownedPendingSeen[key] {
		return true
	}
	ownedPendingSeen[key] = true
	ownedPendingList = append(ownedPendingList, ownedPending{owner: owner, pkg: pkg, source: source, decls: slices.Clone(decls)})
	return true
}

func ownedDeclBasename(path string) string {
	if _, last, ok := strings.CutLast(path, "/"); ok {
		path = last
	}
	if _, last, ok := strings.CutLast(path, `\`); ok {
		path = last
	}
	return path
}

func ownedDeclsKey(decls []logosowned.Declaration) string {
	var b strings.Builder
	for i, d := range decls {
		if i > 0 {
			b.WriteString("\x01")
		}
		b.WriteString(d.Owner)
		b.WriteString("\x00")
		b.WriteString(d.File)
	}
	return b.String()
}

func ownedPendingDeclsKey(decls []logosowned.Declaration) string {
	cp := slices.Clone(decls)
	slices.SortFunc(cp, func(a, b logosowned.Declaration) int {
		if c := strings.Compare(a.Owner, b.Owner); c != 0 {
			return c
		}
		return strings.Compare(a.File, b.File)
	})
	cp = slices.CompactFunc(cp, func(a, b logosowned.Declaration) bool {
		return a.Owner == b.Owner && a.File == b.File
	})
	return ownedDeclsKey(cp)
}

func ownedDeclsForTypes(roots ...types2.Type) ([]logosowned.Declaration, bool) {
	if base.LogosEject == "" {
		return []logosowned.Declaration{}, true
	}
	seen := make(map[types2.Type]bool)
	var out []logosowned.Declaration
	for _, r := range roots {
		if !ownedCollectTypeDecls(r, seen, &out) {
			return nil, false
		}
	}
	if out == nil {
		out = []logosowned.Declaration{}
	}
	return out, true
}

func ownedCollectTypeDecls(t types2.Type, seen map[types2.Type]bool, out *[]logosowned.Declaration) bool {
	if t == nil {
		return true
	}
	if seen[t] {
		return true
	}
	seen[t] = true
	switch u := t.(type) {
	case *types2.Basic:
		return true
	case *types2.TypeParam:
		return true
	case *types2.Named:
		if obj := u.Obj(); obj != nil {
			if pkg := obj.Pkg(); pkg != nil {
				fb := obj.Pos().FileBase()
				if fb == nil {
					return false
				}
				fn := fb.Filename()
				if fn == "" {
					return false
				}
				base := ownedDeclBasename(fn)
				if base == "" || base == "." || base == ".." {
					return false
				}
				if strings.ContainsAny(base, `/\:`) {
					return false
				}
				*out = append(*out, logosowned.Declaration{Owner: pkg.Path(), File: base})
			}
		}
		if args := u.TypeArgs(); args != nil {
			for i := range args.Len() {
				if !ownedCollectTypeDecls(args.At(i), seen, out) {
					return false
				}
			}
		}
		if ul := u.Underlying(); ul != nil {
			if !ownedCollectTypeDecls(ul, seen, out) {
				return false
			}
		}
		return true
	case *types2.Alias:
		if obj := u.Obj(); obj != nil {
			if pkg := obj.Pkg(); pkg != nil {
				fb := obj.Pos().FileBase()
				if fb == nil {
					return false
				}
				fn := fb.Filename()
				if fn == "" {
					return false
				}
				base := ownedDeclBasename(fn)
				if base == "" || base == "." || base == ".." {
					return false
				}
				if strings.ContainsAny(base, `/\:`) {
					return false
				}
				*out = append(*out, logosowned.Declaration{Owner: pkg.Path(), File: base})
			}
		}
		if args := u.TypeArgs(); args != nil {
			for i := range args.Len() {
				if !ownedCollectTypeDecls(args.At(i), seen, out) {
					return false
				}
			}
		}
		if rhs := u.Rhs(); rhs != nil {
			if !ownedCollectTypeDecls(rhs, seen, out) {
				return false
			}
		}
		return true
	case *types2.Pointer:
		return ownedCollectTypeDecls(u.Elem(), seen, out)
	case *types2.Slice:
		return ownedCollectTypeDecls(u.Elem(), seen, out)
	case *types2.Array:
		return ownedCollectTypeDecls(u.Elem(), seen, out)
	case *types2.Map:
		if !ownedCollectTypeDecls(u.Key(), seen, out) {
			return false
		}
		return ownedCollectTypeDecls(u.Elem(), seen, out)
	case *types2.Chan:
		return ownedCollectTypeDecls(u.Elem(), seen, out)
	case *types2.Struct:
		for i := range u.NumFields() {
			f := u.Field(i)
			if f == nil {
				continue
			}
			if !ownedCollectTypeDecls(f.Type(), seen, out) {
				return false
			}
		}
		return true
	case *types2.Signature:
		if ps := u.Params(); ps != nil {
			for i := range ps.Len() {
				if v := ps.At(i); v != nil {
					if !ownedCollectTypeDecls(v.Type(), seen, out) {
						return false
					}
				}
			}
		}
		if rs := u.Results(); rs != nil {
			for i := range rs.Len() {
				if v := rs.At(i); v != nil {
					if !ownedCollectTypeDecls(v.Type(), seen, out) {
						return false
					}
				}
			}
		}
		return true
	case *types2.Interface:
		for i := range u.NumExplicitMethods() {
			m := u.ExplicitMethod(i)
			if m == nil {
				continue
			}
			if !ownedCollectTypeDecls(m.Type(), seen, out) {
				return false
			}
		}
		for i := range u.NumEmbeddeds() {
			if !ownedCollectTypeDecls(u.EmbeddedType(i), seen, out) {
				return false
			}
		}
		return true
	case *types2.Union:
		for i := range u.Len() {
			tm := u.Term(i)
			if tm == nil {
				continue
			}
			if !ownedCollectTypeDecls(tm.Type(), seen, out) {
				return false
			}
		}
		return true
	case *types2.Tuple:
		for i := range u.Len() {
			if v := u.At(i); v != nil {
				if !ownedCollectTypeDecls(v.Type(), seen, out) {
					return false
				}
			}
		}
		return true
	case *types2.ConstView:
		return ownedCollectTypeDecls(u.Elem, seen, out)
	default:
		return true
	}
}

var ownedKnownOS = map[string]bool{"aix": true, "android": true, "darwin": true, "dragonfly": true, "freebsd": true, "hurd": true, "illumos": true, "ios": true, "js": true, "linux": true, "nacl": true, "netbsd": true, "openbsd": true, "plan9": true, "solaris": true, "windows": true, "zos": true}
var ownedKnownArch = map[string]bool{"386": true, "amd64": true, "arm": true, "arm64": true, "loong64": true, "mips": true, "mipsle": true, "mips64": true, "mips64le": true, "ppc64": true, "ppc64le": true, "riscv": true, "riscv64": true, "s390x": true, "sparc64": true, "wasm": true}

func ownedConstraintHeader(fl *fileLower) string {
	if fl == nil || fl.file == nil {
		return ""
	}
	name := fl.baseName
	if name == "" && fl.file.Pos().FileBase() != nil {
		name = fl.file.Pos().FileBase().Filename()
		if i := len(name); i > 0 {
			for j := len(name) - 1; j >= 0; j-- {
				if name[j] == '/' {
					name = name[j+1:]
					break
				}
			}
		}
	}
	if name == "" {
		return ""
	}
	base := strings.TrimSuffix(name, ".go")
	if stem, _, ok := logosfile.Split(name); ok {
		base = stem
	}
	parts := strings.Split(base, "_")
	if len(parts) < 2 {
		return explicitBuildHeader(fl)
	}
	var cond []string
	n := len(parts)
	if n >= 3 && ownedKnownOS[parts[n-2]] && ownedKnownArch[parts[n-1]] {
		cond = append(cond, parts[n-2], parts[n-1])
	} else if ownedKnownOS[parts[n-1]] || ownedKnownArch[parts[n-1]] {
		cond = append(cond, parts[n-1])
	}
	exp := explicitBuildHeader(fl)
	if len(cond) == 0 {
		return exp
	}
	condText := cond[0]
	if len(cond) == 2 {
		condText = "(" + cond[0] + " && " + cond[1] + ")"
	}
	return combineOwnedConstraints(exp, condText)
}

func combineOwnedConstraints(header, filename string) string {
	var modern constraint.Expr
	var legacy constraint.Expr
	for _, line := range strings.Split(header, "\n") {
		if !constraint.IsGoBuild(line) && !constraint.IsPlusBuild(line) {
			continue
		}
		expr, err := constraint.Parse(line)
		if err != nil {
			base.Fatalf("logos: invalid owned helper build constraint: %v", err)
		}
		if constraint.IsGoBuild(line) {
			modern = expr
		} else if legacy == nil {
			legacy = expr
		} else {
			legacy = &constraint.AndExpr{X: legacy, Y: expr}
		}
	}
	expr := modern
	if expr == nil {
		expr = legacy
	}
	if filename != "" {
		fileExpr, err := constraint.Parse("//go:build " + filename)
		if err != nil {
			base.Fatalf("logos: invalid owned filename constraint: %v", err)
		}
		if expr == nil {
			expr = fileExpr
		} else {
			expr = &constraint.AndExpr{X: expr, Y: fileExpr}
		}
	}
	if expr == nil {
		return ""
	}
	lines, err := constraint.PlusBuildLines(expr)
	if err != nil {
		base.Fatalf("logos: cannot render owned legacy constraints: %v", err)
	}
	return "//go:build " + expr.String() + "\n" + strings.Join(lines, "\n") + "\n\n"
}

func explicitBuildHeader(fl *fileLower) string {
	if len(fl.src) == 0 {
		return ""
	}
	text := fl.src
	if len(text) > 4096 {
		text = text[:4096]
	}
	lines := strings.Split(string(text), "\n")
	var out []string
	for _, ln := range lines {
		if len(ln) >= 10 && ln[:10] == "//go:build" {
			out = append(out, ln)
		} else if len(ln) >= 8 && ln[:8] == "// +build" {
			out = append(out, ln)
		}
		if len(out) > 0 && len(ln) == 0 {
			break
		}
		if strings.HasPrefix(strings.TrimLeft(ln, " \t"), "package") {
			break
		}
	}
	if len(out) == 0 {
		return ""
	}
	s := ""
	for _, ln := range out {
		s += ln + "\n"
	}
	return s + "\n"
}

func (fl *fileLower) existingImportName(pkgPath string) (string, bool) {
	for _, decl := range fl.file.DeclList {
		id, ok := decl.(*syntax.ImportDecl)
		if !ok || id.Path == nil {
			continue
		}
		path, _ := strconv.Unquote(id.Path.Value)
		if path == pkgPath {
			if local, ok := fl.cfImportLocalName(id); ok {
				return local, true
			}
		}
	}
	return "", false
}

func ownedFactoryCall(init syntax.Expr, info *types2.Info) (factoryPath, factoryName string, call *syntax.CallExpr, ok bool) {
	call, ok = syntax.Unparen(init).(*syntax.CallExpr)
	if !ok {
		return "", "", nil, false
	}
	fun := syntax.Unparen(call.Fun)
	if indexed, ok := fun.(*syntax.IndexExpr); ok {
		fun = syntax.Unparen(indexed.X)
	}
	sel, ok := fun.(*syntax.SelectorExpr)
	if !ok || sel.Sel == nil || !token.IsExported(sel.Sel.Value) {
		return "", "", nil, false
	}
	var obj types2.Object
	if s, found := info.Selections[sel]; found && s.Obj() != nil {
		obj = s.Obj()
	} else {
		obj = info.Uses[sel.Sel]
	}
	fn, ok := obj.(*types2.Func)
	if !ok || fn.Pkg() == nil {
		return "", "", nil, false
	}
	return fn.Pkg().Path(), sel.Sel.Value, call, true
}

func ownedMapPath(sig *types2.Signature, want types2.LogosOwnedStepKind) ([]types2.LogosOwnedStep, bool) {
	if sig.Results().Len() != 1 {
		return nil, false
	}
	rt := sig.Results().At(0).Type()
	var path []types2.LogosOwnedStep
	path = append(path, types2.LogosOwnedStep{Kind: types2.LogosOwnedSigResult, Index: 0})
	if _, ok := rt.(*types2.Named); ok {
		path = append(path, types2.LogosOwnedStep{Kind: types2.LogosOwnedUnderlying})
		rt = rt.Underlying()
	} else if _, ok := rt.(*types2.Alias); ok {
		path = append(path, types2.LogosOwnedStep{Kind: types2.LogosOwnedUnderlying})
		rt = types2.Unalias(rt)
		if n, ok := rt.(*types2.Named); ok {
			path = append(path, types2.LogosOwnedStep{Kind: types2.LogosOwnedUnderlying})
			rt = n.Underlying()
		}
	}
	if _, ok := rt.Underlying().(*types2.Map); !ok {
		return nil, false
	}
	path = append(path, types2.LogosOwnedStep{Kind: want})
	return path, true
}

func isGenericFactory(fl *fileLower, sig *types2.Signature, init syntax.Expr) bool {
	if sig.TypeParams().Len() > 0 {
		return true
	}
	if call, ok := syntax.Unparen(init).(*syntax.CallExpr); ok {
		found := false
		syntax.Inspect(call.Fun, func(n syntax.Node) bool {
			if name, ok := n.(*syntax.Name); ok {
				if _, ok := fl.pkg.info.Instances[name]; ok {
					found = true
				}
			}
			return true
		})
		return found
	}
	return false
}

func ownedScalarPath(sig *types2.Signature) ([]types2.LogosOwnedStep, bool) {
	if sig.Results().Len() != 1 {
		return nil, false
	}
	return []types2.LogosOwnedStep{{Kind: types2.LogosOwnedSigResult, Index: 0}}, true
}

func (fl *fileLower) ownedScalarBridge(obj *types2.Const, v ctfe.Value, init syntax.Expr, pos syntax.Pos) (ctfeResult, bool) {
	var zero ctfeResult
	fp, fn, sig, anchor, ok := fl.ownedProvenanceDepth(obj, init, 0)
	if !ok {
		return zero, false
	}
	path, ok := ownedScalarPath(sig)
	if !ok {
		return zero, false
	}
	var desc types2.LogosOwnedType
	if isGenericFactory(fl, sig, anchor) {
		desc = types2.LogosOwnedWitness(anchor)
	} else {
		desc = types2.LogosOwnedFactory(fp, fn, path...)
	}
	if ownedWitnessByType == nil {
		ownedWitnessByType = make(map[types2.Type]types2.LogosOwnedType)
	}
	ownedWitnessByType[obj.Type()] = desc
	ownedWitnessByType[ownedUnwrapView(obj.Type())] = desc
	leaf, err := v.Expr(pos)
	if err != nil {
		return zero, false
	}
	typeName := ownedBareTypeName(obj.Type())
	defPath := ownedDefiningPath(obj.Type())
	if defPath == "" {
		defPath = fp
	}
	defPkg := ownedDefPkgName(fl, obj.Type(), defPath)
	if defPkg == "" {
		return zero, false
	}
	phName := ownedNativePlaceholder(fl, "T")
	phNode := nameOf(pos, phName)
	RegisterOwnedNode(phNode, desc)
	lit := at(&syntax.CallExpr{Fun: phNode, ArgList: []syntax.Expr{leaf}}, pos)
	leafText := renderExpr(leaf)
	alias := ownedValueAliasFor(fp, fn, path, "Const", leafText)
	var decl string
	if typeName != "" {
		decl = fmt.Sprintf("package %s\nconst %s = %s(%s)\n", defPkg, alias, typeName, leafText)
	} else {
		decl = fmt.Sprintf("package %s\nconst %s = %s\n", defPkg, alias, leafText)
	}
	decls, dok := ownedDeclsForTypes(obj.Type())
	if !dok {
		return zero, false
	}
	if !fl.emitOwnedRequest(defPath, defPkg, decl, decls) {
		return zero, false
	}
	owner := defPath
	var text string
	if owner == fl.pkg.tpkg.Path() {
		text = alias
	} else {
		q := fl.ensureEjectImport(owner, pos)
		text = q + "." + alias
	}
	return ctfeResult{obj: obj, val: v, init: init, lit: lit, text: text}, true
}

func ownedFactorySig(info *types2.Info, fp, fn string, call *syntax.CallExpr) (*types2.Signature, bool) {
	if call != nil {
		callee := syntax.Unparen(call.Fun)
		for {
			indexed, ok := callee.(*syntax.IndexExpr)
			if !ok {
				break
			}
			callee = syntax.Unparen(indexed.X)
		}
		var name *syntax.Name
		switch callee := callee.(type) {
		case *syntax.Name:
			name = callee
		case *syntax.SelectorExpr:
			name = callee.Sel
		}
		if inst, ok := info.Instances[name]; ok {
			if sig, ok := inst.Type.(*types2.Signature); ok {
				return sig, true
			}
		}
		if sig, ok := info.TypeOf(call.Fun).(*types2.Signature); ok {
			return sig, true
		}
	}
	for _, obj := range info.Defs {
		if f, ok := obj.(*types2.Func); ok && f.Pkg() != nil && f.Pkg().Path() == fp && f.Name() == fn {
			if sig, ok := f.Type().(*types2.Signature); ok {
				return sig, true
			}
		}
	}
	for _, obj := range info.Uses {
		if f, ok := obj.(*types2.Func); ok && f.Pkg() != nil && f.Pkg().Path() == fp && f.Name() == fn {
			if sig, ok := f.Type().(*types2.Signature); ok {
				return sig, true
			}
		}
	}
	return nil, false
}

func ownedUnwrapView(t types2.Type) types2.Type {
	for t != nil {
		if cv, ok := t.(*types2.ConstView); ok {
			t = cv.Elem
			continue
		}
		if a, ok := t.(*types2.Alias); ok {
			t = types2.Unalias(a)
			continue
		}
		break
	}
	return t
}

func ownedTypesMatch(a, b types2.Type) bool {
	if a == nil || b == nil {
		return false
	}
	a = ownedUnwrapView(a)
	b = ownedUnwrapView(b)
	return types2.Identical(a, b)
}

func (fl *fileLower) ownedPriorConstInit(cobj *types2.Const) syntax.Expr {
	var found syntax.Expr
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if found != nil {
			return false
		}
		cd, ok := n.(*syntax.ConstDecl)
		if !ok {
			return true
		}
		if len(cd.NameList) != 1 {
			return true
		}
		if o, ok := fl.pkg.info.Defs[cd.NameList[0]].(*types2.Const); ok && o == cobj {
			init := cd.Values
			if l, ok := init.(*syntax.ListExpr); ok {
				if len(l.ElemList) != 1 {
					return true
				}
				init = l.ElemList[0]
			}
			found = init
			return false
		}
		return true
	})
	return found
}

func (fl *fileLower) ownedProvenanceDepth(obj *types2.Const, init syntax.Expr, depth int) (string, string, *types2.Signature, syntax.Expr, bool) {
	if depth > 8 || init == nil {
		return "", "", nil, nil, false
	}
	if l, ok := init.(*syntax.ListExpr); ok {
		if len(l.ElemList) == 1 {
			init = l.ElemList[0]
		}
	}
	info := fl.pkg.info
	if fp, fn, call, ok := ownedFactoryCall(init, info); ok {
		if sig, ok := ownedFactorySig(info, fp, fn, call); ok {
			return fp, fn, sig, call, true
		}
	}
	var rfp, rfn string
	var rsig *types2.Signature
	var ranchor syntax.Expr
	found := false
	syntax.Inspect(init, func(n syntax.Node) bool {
		if found {
			return false
		}
		if n == init {
			return true
		}
		call, ok := n.(*syntax.CallExpr)
		if !ok {
			return true
		}
		fp2, fn2, _, ok2 := ownedFactoryCall(call, info)
		if !ok2 {
			return true
		}
		sig2, ok2 := ownedFactorySig(info, fp2, fn2, call)
		if !ok2 {
			return true
		}
		if ct := info.TypeOf(call); ct != nil {
			if !ownedTypesMatch(ct, obj.Type()) {
				return true
			}
		} else if sig2.Results().Len() != 1 || !ownedTypesMatch(sig2.Results().At(0).Type(), obj.Type()) {
			return true
		}
		rfp, rfn, rsig, ranchor = fp2, fn2, sig2, call
		found = true
		return false
	})
	if found {
		return rfp, rfn, rsig, ranchor, true
	}
	var names []*syntax.Name
	syntax.Inspect(init, func(n syntax.Node) bool {
		if nm, ok := n.(*syntax.Name); ok {
			names = append(names, nm)
		}
		return true
	})
	for _, nm := range names {
		o, ok := info.Uses[nm]
		if !ok {
			continue
		}
		cobj, ok := o.(*types2.Const)
		if !ok || cobj == obj {
			continue
		}
		if !ownedTypesMatch(cobj.Type(), obj.Type()) {
			continue
		}
		if priorInit := fl.ownedPriorConstInit(cobj); priorInit != nil {
			if fp3, fn3, sig3, _, ok3 := fl.ownedProvenanceDepth(cobj, priorInit, depth+1); ok3 {
				return fp3, fn3, sig3, init, true
			}
		}
	}
	for t, d := range ownedWitnessByType {
		if d.FactoryPath == "" || d.FactoryName == "" {
			continue
		}
		if !ownedTypesMatch(t, obj.Type()) {
			continue
		}
		if sig, ok := ownedFactorySig(info, d.FactoryPath, d.FactoryName, nil); ok {
			return d.FactoryPath, d.FactoryName, sig, init, true
		}
	}
	return "", "", nil, nil, false
}

func ownedPkgName(fl *fileLower, fp string) string {
	if fp == fl.pkg.tpkg.Path() {
		return fl.pkg.tpkg.Name()
	}
	if q, found := fl.existingImportName(fp); found {
		return q
	}
	if i := strings.LastIndex(fp, "/"); i >= 0 {
		return fp[i+1:]
	}
	return fp
}

func ownedDefPkgName(fl *fileLower, t types2.Type, fallbackPath string) string {
	t = ownedUnwrapView(t)
	if n, ok := types2.Unalias(t).(*types2.Named); ok && n.Obj() != nil && n.Obj().Pkg() != nil {
		return n.Obj().Pkg().Name()
	}
	return ownedPkgName(fl, fallbackPath)
}

func (fl *fileLower) ownedCompositeBridge(obj *types2.Const, v ctfe.Value, init syntax.Expr, pos syntax.Pos) (ctfeResult, bool) {
	var zero ctfeResult
	t := ownedUnwrapView(obj.Type())
	lit := BuildMatLiteral(v, t, pos)
	if lit == nil {
		return zero, false
	}
	fp0, fn0, sig0, anchor0, ok0 := fl.ownedProvenanceDepth(obj, init, 0)
	if !ok0 {
		return zero, false
	}
	path := []types2.LogosOwnedStep{{Kind: types2.LogosOwnedSigResult, Index: 0}}
	var desc0 types2.LogosOwnedType
	if isGenericFactory(fl, sig0, anchor0) {
		desc0 = types2.LogosOwnedWitness(anchor0)
	} else {
		desc0 = types2.LogosOwnedFactory(fp0, fn0, path...)
	}
	if ownedWitnessByType == nil {
		ownedWitnessByType = make(map[types2.Type]types2.LogosOwnedType)
	}
	ownedWitnessByType[obj.Type()] = desc0
	ownedWitnessByType[t] = desc0
	ph0 := nameOf(pos, ownedNativePlaceholder(fl, "T"))
	RegisterOwnedNode(ph0, desc0)
	switch cl := lit.(type) {
	case *syntax.CompositeLit:
		cl.Type = ph0
	case *syntax.SliceExpr:
		if bcl, ok := cl.X.(*syntax.CompositeLit); ok {
			bcl.Type = ph0
		}
	}
	RegisterOwnedNode(lit, types2.LogosOwnedType{})
	syntax.Inspect(lit, func(n syntax.Node) bool {
		if ex, ok := n.(syntax.Expr); ok && ex != nil {
			if _, isLit := ex.(*syntax.CompositeLit); isLit {
				RegisterOwnedNode(ex, types2.LogosOwnedType{})
			}
			if _, isSel := ex.(*syntax.SelectorExpr); isSel {
				RegisterOwnedNode(ex, types2.LogosOwnedType{})
			}
		}
		return true
	})
	fp, fn := fp0, fn0
	defPath := ownedDefiningPath(t)
	if defPath == "" {
		defPath = fp
	}
	defPkg := ownedDefPkgName(fl, t, defPath)
	if defPkg == "" {
		return zero, false
	}
	defType := ownedBareTypeName(t)
	if defType == "" {
		return zero, false
	}
	typeAlias := ownedAliasFor(fp, fn, path, "Type")
	vsrc, ok := fl.ownedValueSource(v, t, defPath)
	if !ok {
		return zero, false
	}
	valueFn := ownedValueAliasFor(fp, fn, path, "Value", vsrc)
	decls, dok := ownedDeclsForTypes(t)
	if !dok {
		return zero, false
	}
	aliasSrc := fmt.Sprintf("package %s\ntype %s = %s\n", defPkg, typeAlias, defType)
	if !fl.emitOwnedRequest(defPath, defPkg, aliasSrc, decls) {
		return zero, false
	}
	valueSrc := fmt.Sprintf("package %s\nfunc %s() %s { return %s }\n", defPkg, valueFn, typeAlias, vsrc)
	if !fl.emitOwnedRequest(defPath, defPkg, valueSrc, decls) {
		return zero, false
	}
	var text string
	if defPath == fl.pkg.tpkg.Path() {
		text = valueFn + "()"
	} else {
		q := fl.ensureEjectImport(defPath, pos)
		text = q + "." + valueFn + "()"
	}
	return ctfeResult{obj: obj, val: v, init: init, lit: lit, text: text}, true
}

func ownedBareTypeName(t types2.Type) string {
	t = ownedUnwrapView(t)
	if n, ok := types2.Unalias(t).(*types2.Named); ok && n.Obj() != nil {
		return n.Obj().Name()
	}
	return ""
}

func ownedDefiningPath(t types2.Type) string {
	t = ownedUnwrapView(t)
	if n, ok := types2.Unalias(t).(*types2.Named); ok && n.Obj() != nil && n.Obj().Pkg() != nil {
		return n.Obj().Pkg().Path()
	}
	return ""
}

func (fl *fileLower) ownedValueSource(v ctfe.Value, t types2.Type, ownerPath string) (string, bool) {
	t = ownedUnwrapView(t)
	if dp := ownedDefiningPath(t); dp != "" && dp != ownerPath {
		value, ok := fl.ownedValueSource(v, t, dp)
		if !ok {
			return "", false
		}
		typeName := ownedBareTypeName(t)
		pkgName := ownedDefPkgName(fl, t, dp)
		qualifier := ownedOwnerImport(ownerPath, dp)
		semantic := sha256.Sum256([]byte(dp + "\x00" + types2.TypeString(t, nil) + "\x00" + value))
		fn := "LogosOwnedValue_" + hex.EncodeToString(semantic[:])
		decls, dok := ownedDeclsForTypes(t)
		if !dok {
			return "", false
		}
		if !fl.emitOwnedRequest(dp, pkgName, "package "+pkgName+"\nfunc "+fn+"() "+typeName+" { return "+value+" }\n", decls) {
			return "", false
		}
		return qualifier + "." + fn + "()", true
	}
	ut := types2.Unalias(t)
	if n, ok := ut.(*types2.Named); ok {
		ut = n.Underlying()
	}
	bare := ownedBareTypeName(t)
	if bare == "" {
		bare = types2.TypeString(t, func(p *types2.Package) string {
			if p.Path() == ownerPath {
				return ""
			}
			return p.Name()
		})
	}
	switch u := ut.(type) {
	case *types2.Basic:
		leaf, err := v.Expr(syntax.Pos{})
		if err != nil {
			return "", false
		}
		if bare == "" {
			return renderExpr(leaf), true
		}
		return bare + "(" + renderExpr(leaf) + ")", true
	case *types2.Struct:
		if bare == "" {
			return "", false
		}
		fields := v.Fields()
		if len(fields) != u.NumFields() {
			return "", false
		}
		var b strings.Builder
		b.WriteString(bare)
		b.WriteString("{")
		for i := range fields {
			if i > 0 {
				b.WriteString(", ")
			}
			fs, ok := fl.ownedValueSource(fields[i], u.Field(i).Type(), ownerPath)
			if !ok {
				return "", false
			}
			b.WriteString(fs)
		}
		b.WriteString("}")
		return b.String(), true
	case *types2.Array:
		if bare == "" {
			return "", false
		}
		arr := v.Elems()
		if int64(len(arr)) != u.Len() {
			return "", false
		}
		var b strings.Builder
		b.WriteString(bare)
		b.WriteString("{")
		for i := range arr {
			if i > 0 {
				b.WriteString(", ")
			}
			es, ok := fl.ownedValueSource(arr[i], u.Elem(), ownerPath)
			if !ok {
				return "", false
			}
			b.WriteString(es)
		}
		b.WriteString("}")
		return b.String(), true
	case *types2.Slice:
		backing, lo, length, cap_, ok := v.SliceHeader()
		if !ok {
			return "", false
		}
		elemT := u.Elem()
		if !v.SliceNeedsBacking() {
			vis := backing[lo : lo+length]
			var b strings.Builder
			b.WriteString(bare)
			b.WriteString("{")
			for i := range vis {
				if i > 0 {
					b.WriteString(", ")
				}
				s, ok := fl.ownedValueSource(vis[i], elemT, ownerPath)
				if !ok {
					return "", false
				}
				b.WriteString(s)
			}
			b.WriteString("}")
			return b.String(), true
		}
		acc := backing[lo:cap_]
		var b strings.Builder
		b.WriteString(bare)
		b.WriteString("{")
		for i := range acc {
			if i > 0 {
				b.WriteString(", ")
			}
			s, ok := fl.ownedValueSource(acc[i], elemT, ownerPath)
			if !ok {
				return "", false
			}
			b.WriteString(s)
		}
		b.WriteString("}")
		b.WriteString(fmt.Sprintf("[0:%d:%d]", length, cap_-lo))
		return b.String(), true
	case *types2.Map:
		keys, vals := v.MapEntries()
		var b strings.Builder
		b.WriteString(bare)
		b.WriteString("{")
		for i := range keys {
			if i > 0 {
				b.WriteString(", ")
			}
			ks, ok := fl.ownedValueSource(keys[i], u.Key(), ownerPath)
			if !ok {
				return "", false
			}
			vs, ok := fl.ownedValueSource(vals[i], u.Elem(), ownerPath)
			if !ok {
				return "", false
			}
			b.WriteString(ks + ": " + vs)
		}
		b.WriteString("}")
		return b.String(), true
	}
	if v.IsNilValue() {
		return bare + "(nil)", true
	}
	return "", false
}

func (fl *fileLower) constMapOwnedTexts(g *constMapGen, pos syntax.Pos) (string, string, bool) {
	elem := constMapElem(g.obj.Type())
	mt, ok := elem.Underlying().(*types2.Map)
	if !ok {
		return "", "", false
	}
	if spellableMatType(mt.Key(), fl.pkg.tpkg) && spellableMatType(mt.Elem(), fl.pkg.tpkg) && spellableMatType(elem, fl.pkg.tpkg) {
		return "", "", false
	}
	init := g.init
	if init == nil {
		for _, f := range fl.file.DeclList {
			cd, ok := f.(*syntax.ConstDecl)
			if !ok || len(cd.NameList) != 1 {
				continue
			}
			if o, ok := fl.pkg.info.Defs[cd.NameList[0]].(*types2.Const); ok && o == g.obj {
				init = cd.Values
				break
			}
		}
	}
	fp, fn, sig, anchor, ok := fl.ownedProvenanceDepth(g.obj, init, 0)
	if !ok || fp == "" || fn == "" {
		base.Errorf("%s: logos: private const map %s needs an exported factory witness", pos.String(), g.obj.Name())
		base.ExitIfErrors()
		return "", "", false
	}
	if sig == nil {
		base.Errorf("%s: logos: private const map %s factory %s.%s has no signature", pos.String(), g.obj.Name(), fp, fn)
		base.ExitIfErrors()
		return "", "", false
	}
	keyPath, ok := ownedMapPath(sig, types2.LogosOwnedMapKey)
	if !ok {
		base.Errorf("%s: logos: private const map %s factory result is not a map", pos.String(), g.obj.Name())
		base.ExitIfErrors()
		return "", "", false
	}
	elemPath, ok := ownedMapPath(sig, types2.LogosOwnedMapElem)
	if !ok {
		base.Errorf("%s: logos: private const map %s factory result is not a map", pos.String(), g.obj.Name())
		base.ExitIfErrors()
		return "", "", false
	}
	var keyDesc, elemDesc types2.LogosOwnedType
	if isGenericFactory(fl, sig, anchor) {
		root := constMapElem(fl.pkg.info.TypeOf(anchor))
		if root == nil {
			root = constMapElem(fl.pkg.info.TypeOf(init))
		}
		keyDesc = types2.LogosOwnedWitness(anchor, ownedChildPath(nil, root, types2.LogosOwnedStep{Kind: types2.LogosOwnedMapKey})...)
		elemDesc = types2.LogosOwnedWitness(anchor, ownedChildPath(nil, root, types2.LogosOwnedStep{Kind: types2.LogosOwnedMapElem})...)
	} else {
		keyDesc = types2.LogosOwnedFactory(fp, fn, keyPath...)
		elemDesc = types2.LogosOwnedFactory(fp, fn, elemPath...)
	}
	if ownedWitnessByType == nil {
		ownedWitnessByType = make(map[types2.Type]types2.LogosOwnedType)
	}
	ownedWitnessByType[mt.Key()] = keyDesc
	ownedWitnessByType[mt.Elem()] = elemDesc
	nativeK := ownedNativePlaceholder(fl, "K")
	nativeV := ownedNativePlaceholder(fl, "V")
	keys, vals := g.constMapEntries()
	phDescs := make(map[string]types2.LogosOwnedType)
	phDescs[nativeK] = keyDesc
	phDescs[nativeV] = elemDesc
	nativeKeyLit := make([]string, len(keys))
	nativeValLit := make([]string, len(vals))
	for i := range keys {
		ks, ok := fl.ownedNativeMapValueRec(keys[i], mt.Key(), keyDesc, pos, phDescs)
		if !ok {
			base.Errorf("%s: logos: private const map %s key %d is not representable", pos.String(), g.obj.Name(), i)
			base.ExitIfErrors()
			return "", "", false
		}
		nativeKeyLit[i] = ks
		vs, ok := fl.ownedNativeMapValueRec(vals[i], mt.Elem(), elemDesc, pos, phDescs)
		if !ok {
			base.Errorf("%s: logos: private const map %s value %d is not representable", pos.String(), g.obj.Name(), i)
			base.ExitIfErrors()
			return "", "", false
		}
		nativeValLit[i] = vs
	}
	order := make([]int, len(keys))
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(a, b int) int { return ctfe.CompareKeys(keys[a], keys[b]) })
	sharedEntries := make([]constMapKey, len(keys))
	for i := range keys {
		sharedEntries[i] = constMapKey{hash: constKeyHash(keys[i]), canon: constKeyCanon(keys[i])}
	}
	var sharedLayout *constMapLayout
	sharedIsSwitch := len(keys) <= constMapSwitchMax
	if !sharedIsSwitch {
		l, err := buildConstMap(sharedEntries, constMapSeedBudget(pos))
		if err != nil {
			base.Errorf("%s: %v", pos.String(), err)
			base.ExitIfErrors()
			return "", "", false
		}
		sharedLayout = l
	}
	nativeMapT := "map[" + nativeK + "]" + nativeV
	name := g.name
	bearing := constMapBearing(vals)
	nativeText := emitConstMapDecls(constMapEmission{
		name:        name,
		kT:          nativeK,
		vT:          nativeV,
		mapT:        nativeMapT,
		keyTexts:    nativeKeyLit,
		valTexts:    nativeValLit,
		builderBody: ownedNativeMapLit(nativeKeyLit, nativeValLit, nativeMapT),
		hashExpr:    ownedNativeHashCall(mt.Key(), "_logos_k"),
		order:       order,
		layout:      sharedLayout,
		isSwitch:    sharedIsSwitch,
		bearing:     bearing,
	})
	decls := parseConstMapDecls(nativeText, pos)
	g.nativeDecls = decls
	byName := make(map[string]types2.LogosOwnedType, len(phDescs)+2)
	for k, v := range phDescs {
		byName[k] = v
	}
	byName[nativeK] = keyDesc
	byName[nativeV] = elemDesc
	for _, d := range decls {
		syntax.Inspect(d, func(n syntax.Node) bool {
			if nm, ok := n.(*syntax.Name); ok {
				if nm.Pos() != pos {
					return true
				}
				if desc, found := byName[nm.Value]; found {
					RegisterOwnedNode(nm, desc)
				}
			}
			if lit, ok := n.(*syntax.CompositeLit); ok {
				RegisterOwnedNode(lit, types2.LogosOwnedType{})
			}
			if sel, ok := n.(*syntax.SelectorExpr); ok {
				if sel.Sel != nil && !token.IsExported(sel.Sel.Value) {
					RegisterOwnedNode(sel, types2.LogosOwnedType{})
				}
			}
			return true
		})
	}
	ejectK, ejectV, ejectMapT, ejectKeys, ejectVals, ejectHash, ok := fl.ownedEjectMapSpelling(ownedMapSpec{g: g, mt: mt, keys: keys, vals: vals, fp: fp, fn: fn, keyPath: keyPath, elemPath: elemPath, pos: pos})
	if !ok {
		base.ExitIfErrors()
		return "", "", false
	}
	ejectText := emitConstMapDecls(constMapEmission{
		name:        name,
		kT:          ejectK,
		vT:          ejectV,
		mapT:        ejectMapT,
		keyTexts:    ejectKeys,
		valTexts:    ejectVals,
		builderBody: ownedNativeMapLit(ejectKeys, ejectVals, ejectMapT),
		hashExpr:    ejectHash,
		order:       order,
		layout:      sharedLayout,
		isSwitch:    sharedIsSwitch,
		bearing:     bearing,
	})
	return nativeText, ejectText, true
}

type ownedMapSpec struct {
	g        *constMapGen
	mt       *types2.Map
	keys     []ctfe.Value
	vals     []ctfe.Value
	fp       string
	fn       string
	keyPath  []types2.LogosOwnedStep
	elemPath []types2.LogosOwnedStep
	pos      syntax.Pos
}

func (fl *fileLower) ownedEjectCompositeTexts(t types2.Type, fp, fn string, path []types2.LogosOwnedStep, pos syntax.Pos) (eject, local, owner, pkg, qual string, ok bool) {
	if s, ok := fl.spellLowered(t); ok && matFieldsAssignable(t, fl.pkg.tpkg) {
		return s, s, "", "", "", true
	}
	if def := ownedDefiningPath(t); def != "" {
		bare := ownedBareTypeName(t)
		alias := ownedAliasFor(fp, fn, path, "Type")
		pkgName := ownedDefPkgName(fl, t, def)
		if pkgName == "" {
			return "", "", "", "", "", false
		}
		if bare != "" {
			src := "package " + pkgName + "\ntype " + alias + " = " + bare + "\n"
			decls, dok := ownedDeclsForTypes(t)
			if !dok {
				return "", "", "", "", "", false
			}
			if !fl.emitOwnedRequest(def, pkgName, src, decls) {
				return "", "", "", "", "", false
			}
		}
		if def == fl.pkg.tpkg.Path() {
			return alias, alias, def, pkgName, "", true
		}
		q := fl.ensureEjectImport(def, pos)
		return q + "." + alias, alias, def, pkgName, q, true
	}
	ut := types2.Unalias(t)
	if n, ok := ut.(*types2.Named); ok {
		ut = n.Underlying()
	}
	switch u := ut.(type) {
	case *types2.Array:
		elemPath := ownedChildPath(path, t, types2.LogosOwnedStep{Kind: types2.LogosOwnedArrayElem})
		eEject, eLocal, eOwner, ePkg, eQual, ok := fl.ownedEjectCompositeTexts(u.Elem(), fp, fn, elemPath, pos)
		if !ok || eOwner == "" {
			return "", "", "", "", "", false
		}
		eject = sprintfArrayLen(u.Len(), eEject)
		local = sprintfArrayLen(u.Len(), eLocal)
		return eject, local, eOwner, ePkg, eQual, true
	case *types2.Slice:
		elemPath := ownedChildPath(path, t, types2.LogosOwnedStep{Kind: types2.LogosOwnedSliceElem})
		eEject, eLocal, eOwner, ePkg, eQual, ok := fl.ownedEjectCompositeTexts(u.Elem(), fp, fn, elemPath, pos)
		if !ok || eOwner == "" {
			return "", "", "", "", "", false
		}
		return "[]" + eEject, "[]" + eLocal, eOwner, ePkg, eQual, true
	case *types2.Pointer:
		elemPath := ownedChildPath(path, t, types2.LogosOwnedStep{Kind: types2.LogosOwnedPointerElem})
		eEject, eLocal, eOwner, ePkg, eQual, ok := fl.ownedEjectCompositeTexts(u.Elem(), fp, fn, elemPath, pos)
		if !ok || eOwner == "" {
			return "", "", "", "", "", false
		}
		return "*" + eEject, "*" + eLocal, eOwner, ePkg, eQual, true
	}
	return "", "", "", "", "", false
}

func sprintfArrayLen(n int64, elem string) string {
	return "[" + strconv.Itoa(int(n)) + "]" + elem
}

func (fl *fileLower) ownedEjectMapSpelling(spec ownedMapSpec) (ejectK, ejectV, ejectMapT string, ejectKeys, ejectVals []string, ejectHash string, ok bool) {
	g := spec.g
	mt := spec.mt
	keys := spec.keys
	vals := spec.vals
	fp := spec.fp
	fn := spec.fn
	keyPath := spec.keyPath
	elemPath := spec.elemPath
	pos := spec.pos
	ejectK0, localK, ownerK, pkgK, qualK, ok := fl.ownedEjectCompositeTexts(mt.Key(), fp, fn, keyPath, pos)
	if !ok {
		return "", "", "", nil, nil, "", false
	}
	ejectV0, localV, ownerV, pkgV, qualV, ok := fl.ownedEjectCompositeTexts(mt.Elem(), fp, fn, elemPath, pos)
	if !ok {
		return "", "", "", nil, nil, "", false
	}
	ejectK = ejectK0
	ejectV = ejectV0
	ejectMapT = "map[" + ejectK + "]" + ejectV
	ejectKeys = make([]string, len(keys))
	ejectVals = make([]string, len(vals))
	for i := range keys {
		if _, ok := mt.Key().Underlying().(*types2.Basic); ok {
			leaf, err := keys[i].Expr(pos)
			if err != nil {
				return "", "", "", nil, nil, "", false
			}
			ejectKeys[i] = renderExpr(leaf)
		} else {
			if ownerK == "" {
				base.Errorf("%s: logos: private const map %s key %d needs owner", pos.String(), g.obj.Name(), i)
				base.ExitIfErrors()
				return "", "", "", nil, nil, "", false
			}
			vsrc, ok := fl.ownedValueSource(keys[i], mt.Key(), ownerK)
			if !ok {
				base.Errorf("%s: logos: private const map %s key %d is not representable", pos.String(), g.obj.Name(), i)
				base.ExitIfErrors()
				return "", "", "", nil, nil, "", false
			}
			ck := ownedAliasFor(fp, fn, keyPath, "Value:key:"+strconv.Itoa(i))
			src := "package " + pkgK + "\nfunc " + ck + "() " + localK + " { return " + vsrc + " }\n"
			kdecls, kdok := ownedDeclsForTypes(mt.Key())
			if !kdok {
				base.Errorf("%s: logos: private const map %s key %d has no declaration position", pos.String(), g.obj.Name(), i)
				base.ExitIfErrors()
				return "", "", "", nil, nil, "", false
			}
			if !fl.emitOwnedRequest(ownerK, pkgK, src, kdecls) {
				return "", "", "", nil, nil, "", false
			}
			if qualK != "" {
				ejectKeys[i] = qualK + "." + ck + "()"
			} else {
				ejectKeys[i] = ck + "()"
			}
		}
		if _, ok := mt.Elem().Underlying().(*types2.Basic); ok {
			leaf, err := vals[i].Expr(pos)
			if err != nil {
				return "", "", "", nil, nil, "", false
			}
			ejectVals[i] = renderExpr(leaf)
		} else {
			if ownerV == "" {
				base.Errorf("%s: logos: private const map %s value %d needs owner", pos.String(), g.obj.Name(), i)
				base.ExitIfErrors()
				return "", "", "", nil, nil, "", false
			}
			vsrc, ok := fl.ownedValueSource(vals[i], mt.Elem(), ownerV)
			if !ok {
				base.Errorf("%s: logos: private const map %s value %d is not representable", pos.String(), g.obj.Name(), i)
				base.ExitIfErrors()
				return "", "", "", nil, nil, "", false
			}
			cv := ownedAliasFor(fp, fn, elemPath, "Value:element:"+strconv.Itoa(i))
			src := "package " + pkgV + "\nfunc " + cv + "() " + localV + " { return " + vsrc + " }\n"
			vdecls, vdok := ownedDeclsForTypes(mt.Elem())
			if !vdok {
				base.Errorf("%s: logos: private const map %s value %d has no declaration position", pos.String(), g.obj.Name(), i)
				base.ExitIfErrors()
				return "", "", "", nil, nil, "", false
			}
			if !fl.emitOwnedRequest(ownerV, pkgV, src, vdecls) {
				return "", "", "", nil, nil, "", false
			}
			if qualV != "" {
				ejectVals[i] = qualV + "." + cv + "()"
			} else {
				ejectVals[i] = cv + "()"
			}
		}
	}
	if _, ok := mt.Key().Underlying().(*types2.Basic); ok {
		ejectHash = fl.constKeyHashText(mt.Key(), "_logos_k", pos)
	} else {
		if ownerK == "" {
			return "", "", "", nil, nil, "", false
		}
		hname := ownedAliasFor(fp, fn, keyPath, "Hash")
		hexpr, ok := ownedNativeHashInline(mt.Key(), "_logos_v")
		if !ok {
			base.Errorf("%s: logos: private const map %s key hash is not representable", pos.String(), g.obj.Name())
			base.ExitIfErrors()
			return "", "", "", nil, nil, "", false
		}
		src := "package " + pkgK + "\nimport _logos_rt \"logos/rt\"\nfunc " + hname + "(_logos_v " + localK + ") uint64 { return " + hexpr + " }\n"
		hdecls, hdok := ownedDeclsForTypes(mt.Key())
		if !hdok {
			base.Errorf("%s: logos: private const map %s key hash has no declaration position", pos.String(), g.obj.Name())
			base.ExitIfErrors()
			return "", "", "", nil, nil, "", false
		}
		if !fl.emitOwnedRequest(ownerK, pkgK, src, hdecls) {
			return "", "", "", nil, nil, "", false
		}
		if qualK != "" {
			ejectHash = qualK + "." + hname + "(_logos_k)"
		} else {
			ejectHash = hname + "(_logos_k)"
		}
	}
	return ejectK, ejectV, ejectMapT, ejectKeys, ejectVals, ejectHash, true
}

var ownedEjectImports map[*syntax.File]map[string]string

func (fl *fileLower) ensureEjectImport(pkgPath string, pos syntax.Pos) string {
	if pkgPath == fl.pkg.tpkg.Path() {
		return ""
	}
	if ownedEjectImports == nil {
		ownedEjectImports = make(map[*syntax.File]map[string]string)
	}
	m := ownedEjectImports[fl.file]
	if m == nil {
		m = make(map[string]string)
		ownedEjectImports[fl.file] = m
	}
	if q, ok := fl.existingImportName(pkgPath); ok {
		m[pkgPath] = q
		return q
	}
	if q, ok := m[pkgPath]; ok {
		return q
	}
	q, ok := fl.cfImportQualifier(pkgPath, pos)
	if !ok {
		return ""
	}
	m[pkgPath] = q
	return q
}

func init() {
	ownedTypeBridge = ownedTypeBridgeFunc
	ownedFieldBridge = ownedFieldBridgeFunc
}

func ownedTypeBridgeFunc(fl *fileLower, t types2.Type, pos syntax.Pos) (syntax.Expr, string, bool) {
	if _, ok := fl.spellLowered(t); ok {
		return nil, "", false
	}
	desc, ok := ownedWitnessByType[t]
	if !ok {
		return nil, "", false
	}
	ph := nameOf(pos, ownedNativePlaceholder(fl, "T"))
	RegisterOwnedNode(ph, desc)
	alias := ownedAliasFor(desc.FactoryPath, desc.FactoryName, desc.Path, "Type")
	defPath := ownedDefiningPath(t)
	if defPath == "" {
		return nil, "", false
	}
	bare := ownedBareTypeName(t)
	if bare == "" {
		return nil, "", false
	}
	defPkg := ownedDefPkgName(fl, t, defPath)
	if defPkg == "" {
		return nil, "", false
	}
	src := "package " + defPkg + "\ntype " + alias + " = " + bare + "\n"
	decls, dok := ownedDeclsForTypes(t)
	if !dok {
		return nil, "", false
	}
	if !fl.emitOwnedRequest(defPath, defPkg, src, decls) {
		return nil, "", false
	}
	if defPath == fl.pkg.tpkg.Path() {
		return ph, alias, true
	}
	q := fl.ensureEjectImport(defPath, pos)
	return ph, q + "." + alias, true
}

func ownedFieldBridgeFunc(fl *fileLower, e *syntax.OptSelectExpr, defNative syntax.Expr, defEject string) (syntax.Expr, string, bool) {
	if e == nil || e.Sel == nil || token.IsExported(e.Sel.Value) {
		return nil, "", false
	}
	if sel, ok := defNative.(*syntax.SelectorExpr); ok {
		RegisterOwnedNode(sel, types2.LogosOwnedType{})
		return defNative, defEject, true
	}
	return nil, "", false
}

func ownedChildPath(parent []types2.LogosOwnedStep, t types2.Type, step types2.LogosOwnedStep) []types2.LogosOwnedStep {
	path := slices.Clone(parent)
	for {
		switch u := t.(type) {
		case *types2.Alias:
			path = append(path, types2.LogosOwnedStep{Kind: types2.LogosOwnedUnderlying})
			t = types2.Unalias(u)
		case *types2.Named:
			path = append(path, types2.LogosOwnedStep{Kind: types2.LogosOwnedUnderlying})
			t = u.Underlying()
		default:
			return append(path, step)
		}
	}
}

func (fl *fileLower) ownedNativeMapValueRec(v ctfe.Value, t types2.Type, parent types2.LogosOwnedType, pos syntax.Pos, phDescs map[string]types2.LogosOwnedType) (string, bool) {
	if v.IsNilValue() {
		ph := ownedNativePlaceholder(fl, "Nil")
		phDescs[ph] = parent
		return ph + "(nil)", true
	}
	if _, ok := types2.Unalias(t).Underlying().(*types2.Basic); ok {
		leaf, err := v.Expr(syntax.Pos{})
		if err != nil {
			return "", false
		}
		return renderExpr(leaf), true
	}
	ut := types2.Unalias(t)
	if n, ok := ut.(*types2.Named); ok {
		ut = n.Underlying()
	}
	desc := parent
	phName := ownedNativePlaceholder(fl, "T")
	if desc.Witness != nil {
		desc = types2.LogosOwnedWitness(desc.Witness, desc.Path...)
	} else {
		desc = types2.LogosOwnedFactory(desc.FactoryPath, desc.FactoryName, desc.Path...)
	}
	phDescs[phName] = desc
	placeholder := phName
	switch u := ut.(type) {
	case *types2.Struct:
		fields := v.Fields()
		if len(fields) != u.NumFields() {
			return "", false
		}
		var b strings.Builder
		b.WriteString(placeholder)
		b.WriteString("{")
		for i := range fields {
			if i > 0 {
				b.WriteString(", ")
			}
			childParent := desc
			childParent.Path = ownedChildPath(desc.Path, t, types2.LogosOwnedStep{Kind: types2.LogosOwnedStructField, Index: i})
			fs, ok := fl.ownedNativeMapValueRec(fields[i], u.Field(i).Type(), childParent, pos, phDescs)
			if !ok {
				return "", false
			}
			b.WriteString(fs)
		}
		b.WriteString("}")
		return b.String(), true
	case *types2.Array:
		arr := v.Elems()
		if int64(len(arr)) != u.Len() {
			return "", false
		}
		var b strings.Builder
		b.WriteString(placeholder)
		b.WriteString("{")
		for i := range arr {
			if i > 0 {
				b.WriteString(", ")
			}
			childParent := desc
			childParent.Path = ownedChildPath(desc.Path, t, types2.LogosOwnedStep{Kind: types2.LogosOwnedArrayElem})
			es, ok := fl.ownedNativeMapValueRec(arr[i], u.Elem(), childParent, pos, phDescs)
			if !ok {
				return "", false
			}
			b.WriteString(es)
		}
		b.WriteString("}")
		return b.String(), true
	case *types2.Slice:
		backing, lo, length, cap_, ok := v.SliceHeader()
		if !ok {
			return "", false
		}
		if v.SliceNeedsBacking() {
			acc := backing[lo:cap_]
			var bb strings.Builder
			bb.WriteString(placeholder)
			bb.WriteString("{")
			for i := range acc {
				if i > 0 {
					bb.WriteString(", ")
				}
				childParent := desc
				childParent.Path = ownedChildPath(desc.Path, t, types2.LogosOwnedStep{Kind: types2.LogosOwnedSliceElem})
				es, ok := fl.ownedNativeMapValueRec(acc[i], u.Elem(), childParent, pos, phDescs)
				if !ok {
					return "", false
				}
				bb.WriteString(es)
			}
			bb.WriteString("}")
			bb.WriteString("[0:" + strconv.Itoa(length) + ":" + strconv.Itoa(cap_-lo) + "]")
			return bb.String(), true
		}
		vis := backing[lo : lo+length]
		var b strings.Builder
		b.WriteString(placeholder)
		b.WriteString("{")
		for i := range vis {
			if i > 0 {
				b.WriteString(", ")
			}
			childParent := desc
			childParent.Path = ownedChildPath(desc.Path, t, types2.LogosOwnedStep{Kind: types2.LogosOwnedSliceElem})
			s, ok := fl.ownedNativeMapValueRec(vis[i], u.Elem(), childParent, pos, phDescs)
			if !ok {
				return "", false
			}
			b.WriteString(s)
		}
		b.WriteString("}")
		return b.String(), true
	case *types2.Map:
		keys, vals := v.MapEntries()
		var b strings.Builder
		b.WriteString(placeholder)
		b.WriteString("{")
		for i := range keys {
			if i > 0 {
				b.WriteString(", ")
			}
			kParent := desc
			kParent.Path = ownedChildPath(desc.Path, t, types2.LogosOwnedStep{Kind: types2.LogosOwnedMapKey})
			vParent := desc
			vParent.Path = ownedChildPath(desc.Path, t, types2.LogosOwnedStep{Kind: types2.LogosOwnedMapElem})
			ks, ok := fl.ownedNativeMapValueRec(keys[i], u.Key(), kParent, pos, phDescs)
			if !ok {
				return "", false
			}
			vs, ok := fl.ownedNativeMapValueRec(vals[i], u.Elem(), vParent, pos, phDescs)
			if !ok {
				return "", false
			}
			b.WriteString(ks + ": " + vs)
		}
		b.WriteString("}")
		return b.String(), true
	}
	leaf, err := v.Expr(syntax.Pos{})
	if err == nil {
		return renderExpr(leaf), true
	}
	return "", false
}

func ownedNativeMapLit(keys, vals []string, mapT string) string {
	var b strings.Builder
	b.WriteString(mapT)
	b.WriteString("{")
	for i := range keys {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(keys[i] + ": " + vals[i])
	}
	b.WriteString("}")
	return b.String()
}

func ownedNativeHashInline(t types2.Type, x string) (string, bool) {
	ut := types2.Unalias(t)
	if n, ok := ut.(*types2.Named); ok {
		ut = n.Underlying()
	}
	switch u := ut.(type) {
	case *types2.Basic:
		info := u.Info()
		switch {
		case info&types2.IsString != 0:
			return "_logos_rt.HashString(string(" + x + "))", true
		case info&types2.IsBoolean != 0:
			return "_logos_rt.HashBool(bool(" + x + "))", true
		case info&types2.IsInteger != 0:
			return "_logos_rt.HashUint64(uint64(" + x + "))", true
		case info&types2.IsFloat != 0:
			return "_logos_rt.HashFloat64(float64(" + x + "))", true
		}
		return "", false
	case *types2.Struct:
		var b strings.Builder
		b.WriteString("func() uint64 { h := uint64(0); ")
		for i := range u.NumFields() {
			f := u.Field(i)
			if f.Name() == "_" {
				continue
			}
			fe, ok := ownedNativeHashInline(f.Type(), x+"."+f.Name())
			if !ok {
				return "", false
			}
			b.WriteString("h = _logos_rt.HashCombine(h, " + fe + "); ")
		}
		b.WriteString("return h }()")
		return b.String(), true
	case *types2.Array:
		ee, ok := ownedNativeHashInline(u.Elem(), "e")
		if !ok {
			return "", false
		}
		return "func() uint64 { h := uint64(0); for _, e := range " + x + " { h = _logos_rt.HashCombine(h, " + ee + ") }; return h }()", true
	}
	return "", false
}

func ownedNativeHashCall(t types2.Type, x string) string {
	expr, ok := ownedNativeHashInline(t, x)
	if !ok {
		base.Fatalf("logos: private const map key hash is not representable")
	}
	return expr
}
