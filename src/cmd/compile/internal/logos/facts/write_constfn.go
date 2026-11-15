// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts

import (
	"cmp"
	"fmt"
	"go/scanner"
	"go/token"
	"maps"
	"slices"
	"strconv"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func writeConstFuncs(pkg *types2.Package, info *types2.Info, files []*syntax.File, src map[*syntax.File][]byte, f *Facts) {
	if pkg == nil || pkg.Scope() == nil || info == nil {
		return
	}
	var roots []*syntax.ConstFuncDecl
	for _, file := range files {
		if _, ok := src[file]; !ok {
			continue
		}
		for _, decl := range file.DeclList {
			cfd, ok := decl.(*syntax.ConstFuncDecl)
			if !ok || cfd.Func == nil || cfd.Func.Name == nil {
				continue
			}
			obj, _ := info.Defs[cfd.Func.Name].(*types2.Func)
			if obj == nil || !obj.Exported() || obj.Pkg() != pkg {
				continue
			}
			sig, ok := obj.Type().(*types2.Signature)
			if !ok || sig.Recv() != nil {
				continue
			}
			if tp := sig.TypeParams(); tp != nil && tp.Len() > 0 {
				continue
			}
			roots = append(roots, cfd)
		}
	}
	if len(roots) == 0 {
		return
	}
	for _, cfd := range roots {
		payload, err := constFnClosure(pkg, info, files, src, cfd)
		if err != nil {
			base.Errorf("%s: logos: cannot close const function %s: %v", cfd.Func.Pos().String(), cfd.Func.Name.Value, err)
			continue
		}
		f.ConstFuncs = append(f.ConstFuncs, ConstFuncFact{Object: cfd.Func.Name.Value, Source: payload})
	}
}

func constFnClosure(pkg *types2.Package, info *types2.Info, files []*syntax.File, src map[*syntax.File][]byte, cfd *syntax.ConstFuncDecl) ([]byte, error) {
	rootFile := fileOf(files, cfd.Func.Pos())
	if rootFile == nil {
		return nil, fmt.Errorf("root declaration's file not found")
	}
	s := newConstFnSrc(pkg, info, files, src)
	s.indexAll()

	rootObj, ok := info.Defs[cfd.Func.Name].(*types2.Func)
	if !ok || rootObj == nil {
		return nil, fmt.Errorf("const function %s has no checked object", cfd.Func.Name.Value)
	}
	seen := map[types2.Object]bool{rootObj: true}
	if err := s.collectDecl(rootFile, syntax.Node(cfd.Func), seen); err != nil {
		return nil, err
	}
	s.finalizeImports()

	rootText, err := s.renderConstFunc(rootFile, cfd)
	if err != nil {
		return nil, err
	}
	funcs := []constFnPiece{{name: cfd.Func.Name.Value, text: rootText, pos: cfd.ConstPos}}

	for _, entry := range s.collectedFuncs {
		var text string
		var pos syntax.Pos
		switch d := entry.decl.(type) {
		case *syntax.ConstFuncDecl:
			t, err := s.renderConstFunc(entry.file, d)
			if err != nil {
				return nil, err
			}
			text, pos = t, d.ConstPos
		case *syntax.FuncDecl:
			t, err := s.renderPlainFunc(entry.file, d)
			if err != nil {
				return nil, err
			}
			text, pos = t, d.Pos()
		default:
			return nil, fmt.Errorf("unexpected function declaration %T", entry.decl)
		}
		funcs = append(funcs, constFnPiece{name: entry.name, text: text, pos: pos})
	}

	var typesList []constFnPiece
	for _, entry := range s.collectedTypes {
		td, ok := entry.decl.(*syntax.TypeDecl)
		if !ok || td == nil {
			return nil, fmt.Errorf("unexpected type declaration %T", entry.decl)
		}
		t, err := s.renderType(entry.file, td)
		if err != nil {
			return nil, err
		}
		typesList = append(typesList, constFnPiece{name: entry.name, text: t, pos: td.Pos()})
	}

	var constsList []constFnPiece
	for _, entry := range s.collectedConsts {
		cd, ok := entry.decl.(*syntax.ConstDecl)
		if !ok || cd == nil {
			return nil, fmt.Errorf("unexpected const declaration %T", entry.decl)
		}
		t, err := s.renderConst(entry.file, cd)
		if err != nil {
			return nil, err
		}
		constsList = append(constsList, constFnPiece{name: entry.name, text: t, pos: cd.Pos()})
	}

	var varsList []constFnPiece
	for _, entry := range s.collectedVars {
		vd, ok := entry.decl.(*syntax.VarDecl)
		if !ok || vd == nil {
			return nil, fmt.Errorf("unexpected var declaration %T", entry.decl)
		}
		t, err := s.renderVar(entry.file, vd, entry.name)
		if err != nil {
			return nil, err
		}
		varsList = append(varsList, constFnPiece{name: entry.name, text: t, pos: vd.Pos()})
	}

	sortPieces(funcs[1:])
	sortPieces(typesList)
	sortPieces(constsList)
	sortPieces(varsList)

	var b strings.Builder
	b.WriteString("package " + rootFile.PkgName.Value + "\n\n")

	imps := slices.Clone(s.imports)
	slices.SortFunc(imps, func(a, b constFnImport) int { return strings.Compare(a.path, b.path) })
	for _, imp := range imps {
		b.WriteString("import " + imp.name + " " + strconv.Quote(imp.path) + "\n")
	}
	if len(imps) > 0 {
		b.WriteString("\n")
	}

	writePieces(&b, funcs)
	writePieces(&b, typesList)
	writePieces(&b, constsList)
	writePieces(&b, varsList)

	return []byte(b.String()), nil
}

type constFnPiece struct {
	name string
	text string
	pos  syntax.Pos
}

func sortPieces(list []constFnPiece) {
	slices.SortFunc(list, func(a, b constFnPiece) int {
		if c := cmp.Compare(a.pos.FileBase().Filename(), b.pos.FileBase().Filename()); c != 0 {
			return c
		}
		if c := cmp.Compare(a.pos.Line(), b.pos.Line()); c != 0 {
			return c
		}
		return cmp.Compare(a.pos.Col(), b.pos.Col())
	})
}

func writePieces(b *strings.Builder, list []constFnPiece) {
	for _, p := range list {
		b.WriteString(p.text)
		b.WriteString("\n\n")
	}
}

type constFnDecl struct {
	name string
	decl syntax.Decl
	file *syntax.File
}

type pkgUse struct {
	pos  syntax.Pos
	len  int
	path string
	file *syntax.File
}

type constFnSrc struct {
	pkg             *types2.Package
	info            *types2.Info
	files           []*syntax.File
	src             map[*syntax.File][]byte
	starts          map[*syntax.File][]int
	keywords        map[*syntax.File]*KeywordIndex
	decls           map[string]*constFnDecl
	rawImports      map[string]string
	imports         []constFnImport
	pkgUses         []pkgUse
	closedNames     map[string]bool
	collectedFuncs  []*constFnDecl
	collectedTypes  []*constFnDecl
	collectedConsts []*constFnDecl
	collectedVars   []*constFnDecl
	file            *syntax.File
	orig            []byte
	curStarts       []int
}

func newConstFnSrc(pkg *types2.Package, info *types2.Info, files []*syntax.File, src map[*syntax.File][]byte) *constFnSrc {
	starts := make(map[*syntax.File][]int, len(files))
	for _, f := range files {
		if data, ok := src[f]; ok && data != nil {
			starts[f] = lineStarts(data)
		}
	}
	return &constFnSrc{
		pkg:         pkg,
		info:        info,
		files:       files,
		src:         src,
		starts:      starts,
		keywords:    make(map[*syntax.File]*KeywordIndex),
		decls:       make(map[string]*constFnDecl),
		rawImports:  make(map[string]string),
		closedNames: make(map[string]bool),
	}
}

func (s *constFnSrc) indexAll() {
	for _, file := range s.files {
		if _, ok := s.src[file]; !ok {
			continue
		}
		for _, decl := range file.DeclList {
			switch decl := decl.(type) {
			case *syntax.FuncDecl:
				if decl.Name != nil && decl.Recv == nil {
					s.decls[decl.Name.Value] = &constFnDecl{name: decl.Name.Value, decl: decl, file: file}
				}
			case *syntax.ConstFuncDecl:
				if decl.Func != nil && decl.Func.Name != nil && decl.Func.Recv == nil {
					s.decls[decl.Func.Name.Value] = &constFnDecl{name: decl.Func.Name.Value, decl: decl, file: file}
				}
			case *syntax.TypeDecl:
				if decl.Name != nil {
					s.decls[decl.Name.Value] = &constFnDecl{name: decl.Name.Value, decl: decl, file: file}
				}
			case *syntax.ConstDecl:
				for _, n := range decl.NameList {
					if n != nil {
						s.decls[n.Value] = &constFnDecl{name: n.Value, decl: decl, file: file}
					}
				}
			case *syntax.VarDecl:
				for _, n := range decl.NameList {
					if n != nil {
						s.decls[n.Value] = &constFnDecl{name: n.Value, decl: decl, file: file}
					}
				}
			}
		}
	}
}

func (s *constFnSrc) at(file *syntax.File) bool {
	orig, ok := s.src[file]
	if !ok || orig == nil {
		return false
	}
	s.file = file
	s.orig = orig
	s.curStarts = s.starts[file]
	return true
}

func (s *constFnSrc) off(pos syntax.Pos) int {
	return s.curStarts[pos.Line()-1] + int(pos.Col()) - 1
}

func (s *constFnSrc) collectDecl(file *syntax.File, root syntax.Node, seen map[types2.Object]bool) error {
	var firstErr error
	syntax.Inspect(root, func(n syntax.Node) bool {
		if firstErr != nil {
			return false
		}
		name, ok := n.(*syntax.Name)
		if !ok {
			return true
		}
		s.closedNames[name.Value] = true
		obj := s.info.Uses[name]
		if obj == nil {
			return true
		}
		if pn, ok := obj.(*types2.PkgName); ok {
			s.pkgUses = append(s.pkgUses, pkgUse{
				pos:  name.Pos(),
				len:  len(name.Value),
				path: pn.Imported().Path(),
				file: file,
			})
			if err := s.addImport(pn); err != nil && firstErr == nil {
				firstErr = err
				return false
			}
			return true
		}
		if obj.Pkg() != s.pkg {
			return true
		}
		if s.pkg.Scope().Lookup(obj.Name()) != obj {
			return true
		}
		if seen[obj] {
			return true
		}
		if err := s.collectObject(obj, seen); err != nil && firstErr == nil {
			firstErr = err
			return false
		}
		return true
	})
	return firstErr
}

func (s *constFnSrc) collectObject(obj types2.Object, seen map[types2.Object]bool) error {
	if obj == nil || obj.Pkg() != s.pkg || seen[obj] {
		return nil
	}
	if s.pkg.Scope().Lookup(obj.Name()) != obj {
		return nil
	}
	switch obj := obj.(type) {
	case *types2.Func:
		seen[obj] = true
		entry, ok := s.decls[obj.Name()]
		if !ok || entry == nil {
			return fmt.Errorf("declaration of same-package function %s not found", obj.Name())
		}
		s.collectedFuncs = append(s.collectedFuncs, entry)
		var inner *syntax.FuncDecl
		switch d := entry.decl.(type) {
		case *syntax.ConstFuncDecl:
			inner = d.Func
		case *syntax.FuncDecl:
			inner = d
		}
		return s.collectDecl(entry.file, syntax.Node(inner), seen)

	case *types2.TypeName:
		seen[obj] = true
		entry, ok := s.decls[obj.Name()]
		if !ok || entry == nil {
			return fmt.Errorf("declaration of same-package type %s not found", obj.Name())
		}
		td, ok := entry.decl.(*syntax.TypeDecl)
		if !ok || td == nil {
			return fmt.Errorf("declaration of same-package type %s not found", obj.Name())
		}
		if td.Group != nil {
			for _, spec := range s.groupTypeSpecs(entry.file, td.Group) {
				if spec.Name != nil {
					if o := s.info.Defs[spec.Name]; o != nil {
						seen[o] = true
					}
				}
			}
		}
		s.collectedTypes = append(s.collectedTypes, entry)
		if td.Group != nil {
			for _, spec := range s.groupTypeSpecs(entry.file, td.Group) {
				if spec.Type != nil {
					if err := s.collectDecl(entry.file, syntax.Node(spec.Type), seen); err != nil {
						return err
					}
				}
			}
			return nil
		}
		return s.collectDecl(entry.file, syntax.Node(td.Type), seen)

	case *types2.Const:
		seen[obj] = true
		entry, ok := s.decls[obj.Name()]
		if !ok || entry == nil {
			return fmt.Errorf("declaration of same-package constant %s not found", obj.Name())
		}
		cd, ok := entry.decl.(*syntax.ConstDecl)
		if !ok || cd == nil {
			return fmt.Errorf("declaration of same-package constant %s not found", obj.Name())
		}
		for _, n := range cd.NameList {
			if n != nil {
				if o := s.info.Defs[n]; o != nil {
					seen[o] = true
				}
			}
		}
		if cd.Group != nil {
			for _, spec := range s.groupConstSpecs(entry.file, cd.Group) {
				for _, n := range spec.NameList {
					if n != nil {
						if o := s.info.Defs[n]; o != nil {
							seen[o] = true
						}
					}
				}
			}
		}
		s.collectedConsts = append(s.collectedConsts, entry)
		if cd.Group != nil {
			for _, spec := range s.groupConstSpecs(entry.file, cd.Group) {
				if spec.Values != nil {
					if err := s.collectDecl(entry.file, syntax.Node(spec.Values), seen); err != nil {
						return err
					}
				}
				if spec.Type != nil {
					if err := s.collectDecl(entry.file, syntax.Node(spec.Type), seen); err != nil {
						return err
					}
				}
			}
			return nil
		}
		if cd.Values != nil {
			if err := s.collectDecl(entry.file, syntax.Node(cd.Values), seen); err != nil {
				return err
			}
		}
		if cd.Type != nil {
			if err := s.collectDecl(entry.file, syntax.Node(cd.Type), seen); err != nil {
				return err
			}
		}
		return nil

	case *types2.Var:
		seen[obj] = true
		entry, ok := s.decls[obj.Name()]
		if !ok || entry == nil {
			return fmt.Errorf("declaration of same-package variable %s not found", obj.Name())
		}
		vd, ok := entry.decl.(*syntax.VarDecl)
		if !ok || vd == nil {
			return fmt.Errorf("declaration of same-package variable %s not found", obj.Name())
		}
		s.collectedVars = append(s.collectedVars, entry)
		if vd.Type != nil {
			return s.collectDecl(entry.file, syntax.Node(vd.Type), seen)
		}
		return s.collectNamedTypes(obj.Type(), seen)
	}
	return nil
}

func (s *constFnSrc) collectNamedTypes(t types2.Type, seen map[types2.Object]bool) error {
	if t == nil {
		return nil
	}
	if a, ok := t.(*types2.Alias); ok {
		if a.Obj() != nil {
			if opkg := a.Obj().Pkg(); opkg != nil && opkg != s.pkg {
				if err := s.recordImport(opkg.Path(), opkg.Name()); err != nil {
					return err
				}
			} else if opkg == s.pkg {
				if err := s.collectObject(a.Obj(), seen); err != nil {
					return err
				}
			}
		}
		if ta := a.TypeArgs(); ta != nil {
			for i := range ta.Len() {
				if err := s.collectNamedTypes(ta.At(i), seen); err != nil {
					return err
				}
			}
		}
		return s.collectNamedTypes(a.Rhs(), seen)
	}
	switch t := types2.Unalias(t).(type) {
	case *types2.Named:
		if t.Obj() != nil {
			if opkg := t.Obj().Pkg(); opkg != nil && opkg != s.pkg {
				if err := s.recordImport(opkg.Path(), opkg.Name()); err != nil {
					return err
				}
			} else if opkg == s.pkg {
				if err := s.collectObject(t.Obj(), seen); err != nil {
					return err
				}
			}
		}
		if tp := t.TypeArgs(); tp != nil {
			for i := range tp.Len() {
				if err := s.collectNamedTypes(tp.At(i), seen); err != nil {
					return err
				}
			}
		}
	case *types2.Pointer:
		return s.collectNamedTypes(t.Elem(), seen)
	case *types2.Slice:
		return s.collectNamedTypes(t.Elem(), seen)
	case *types2.Array:
		return s.collectNamedTypes(t.Elem(), seen)
	case *types2.Map:
		if err := s.collectNamedTypes(t.Key(), seen); err != nil {
			return err
		}
		return s.collectNamedTypes(t.Elem(), seen)
	case *types2.Chan:
		return s.collectNamedTypes(t.Elem(), seen)
	case *types2.Struct:
		for i := range t.NumFields() {
			if err := s.collectNamedTypes(t.Field(i).Type(), seen); err != nil {
				return err
			}
		}
	case *types2.Tuple:
		for i := range t.Len() {
			if err := s.collectNamedTypes(t.At(i).Type(), seen); err != nil {
				return err
			}
		}
	case *types2.Signature:
		if t.Params() != nil {
			if err := s.collectNamedTypes(t.Params(), seen); err != nil {
				return err
			}
		}
		if t.Results() != nil {
			if err := s.collectNamedTypes(t.Results(), seen); err != nil {
				return err
			}
		}
	case *types2.Interface:
		for i := range t.NumExplicitMethods() {
			if err := s.collectNamedTypes(t.ExplicitMethod(i).Type(), seen); err != nil {
				return err
			}
		}
		for i := range t.NumEmbeddeds() {
			if err := s.collectNamedTypes(t.EmbeddedType(i), seen); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *constFnSrc) addImport(pn *types2.PkgName) error {
	if pn == nil || pn.Imported() == nil {
		return fmt.Errorf("import of unnamed package")
	}
	return s.recordImport(pn.Imported().Path(), pn.Imported().Name())
}

func (s *constFnSrc) recordImport(path, name string) error {
	if name == "" || name == "." || name == "_" {
		return fmt.Errorf("import of %s binds no usable package name", path)
	}
	if _, ok := s.rawImports[path]; !ok {
		s.rawImports[path] = name
	}
	return nil
}

func (s *constFnSrc) finalizeImports() {
	paths := slices.Sorted(maps.Keys(s.rawImports))
	for _, path := range paths {
		baseName := s.rawImports[path]
		name := baseName
		if !s.freshName(name) {
			for i := 1; ; i++ {
				cand := name + strconv.Itoa(i)
				if s.freshName(cand) {
					name = cand
					break
				}
			}
		}
		s.imports = append(s.imports, constFnImport{name: name, path: path})
	}
}

func (s *constFnSrc) aliasFor(path string) string {
	for _, imp := range s.imports {
		if imp.path == path {
			return imp.name
		}
	}
	return ""
}

func (s *constFnSrc) freshName(name string) bool {
	for _, imp := range s.imports {
		if imp.name == name {
			return false
		}
	}
	if s.closedNames[name] {
		return false
	}
	for _, file := range s.files {
		if slices.ContainsFunc(file.DeclList, func(d syntax.Decl) bool {
			return declDeclaresName(d, name)
		}) {
			return false
		}
	}
	return true
}

func declDeclaresName(d syntax.Decl, name string) bool {
	switch d := d.(type) {
	case *syntax.FuncDecl:
		return d.Name != nil && d.Name.Value == name
	case *syntax.ConstFuncDecl:
		return d.Func != nil && d.Func.Name != nil && d.Func.Name.Value == name
	case *syntax.TypeDecl:
		return d.Name != nil && d.Name.Value == name
	case *syntax.ConstDecl:
		return slices.ContainsFunc(d.NameList, func(n *syntax.Name) bool { return n != nil && n.Value == name })
	case *syntax.VarDecl:
		return slices.ContainsFunc(d.NameList, func(n *syntax.Name) bool { return n != nil && n.Value == name })
	case *syntax.ImportDecl:
		if d.LocalPkgName != nil {
			return d.LocalPkgName.Value == name
		}
		if d.Path != nil {
			p := unquotePath(d.Path.Value)
			if _, last, ok := strings.CutLast(p, "/"); ok {
				return last == name
			}
			return p == name
		}
	}
	return false
}

func unquotePath(lit string) string {
	if len(lit) >= 2 && lit[0] == '"' && lit[len(lit)-1] == '"' {
		return lit[1 : len(lit)-1]
	}
	return lit
}

func (s *constFnSrc) renderConstFunc(file *syntax.File, cfd *syntax.ConstFuncDecl) (string, error) {
	if !s.at(file) {
		return "", fmt.Errorf("cannot load file for const function %s", cfd.Func.Name.Value)
	}
	start := s.off(cfd.ConstPos)
	end := s.funcEnd(cfd.Func)
	if end <= start || end > len(s.orig) {
		return "", fmt.Errorf("span out of bounds for const function %s", cfd.Func.Name.Value)
	}
	return s.spliceRange(file, start, end)
}

func (s *constFnSrc) renderPlainFunc(file *syntax.File, fd *syntax.FuncDecl) (string, error) {
	if !s.at(file) {
		return "", fmt.Errorf("cannot load file for function %s", fd.Name.Value)
	}
	start, ok := s.keywordStart(token.FUNC, s.off(syntax.StartPos(fd)))
	if !ok {
		return "", fmt.Errorf("cannot find func keyword for %s", fd.Name.Value)
	}
	end := s.funcEnd(fd)
	if end <= start || end > len(s.orig) {
		return "", fmt.Errorf("span out of bounds for function %s", fd.Name.Value)
	}
	return s.spliceRange(file, start, end)
}

func (s *constFnSrc) renderType(file *syntax.File, td *syntax.TypeDecl) (string, error) {
	if !s.at(file) {
		return "", fmt.Errorf("cannot load file for type %s", td.Name.Value)
	}
	start, ok := s.typeStart(td)
	if !ok {
		return "", fmt.Errorf("cannot find type keyword for %s", td.Name.Value)
	}
	end := s.typeEnd(td)
	if end <= start || end > len(s.orig) {
		return "", fmt.Errorf("span out of bounds for type %s", td.Name.Value)
	}
	return s.spliceRange(file, start, end)
}

func (s *constFnSrc) renderConst(file *syntax.File, cd *syntax.ConstDecl) (string, error) {
	if !s.at(file) {
		return "", fmt.Errorf("cannot load file for const %s", cd.NameList[0].Value)
	}
	start, ok := s.constStart(cd)
	if !ok {
		return "", fmt.Errorf("cannot find const keyword for %s", cd.NameList[0].Value)
	}
	end := s.constEnd(cd)
	if end <= start || end > len(s.orig) {
		return "", fmt.Errorf("span out of bounds for const %s", cd.NameList[0].Value)
	}
	return s.spliceRange(file, start, end)
}

func (s *constFnSrc) renderVar(file *syntax.File, vd *syntax.VarDecl, name string) (string, error) {
	s.at(file)
	if vd.Type != nil {
		start := s.off(syntax.StartPos(vd.Type))
		end := s.typeExprEnd(vd.Type)
		if end <= start || end > len(s.orig) {
			return "", fmt.Errorf("span out of bounds for var %s type", name)
		}
		typeText, err := s.spliceRange(file, start, end)
		if err != nil {
			return "", err
		}
		return "var " + name + " " + typeText, nil
	}
	obj := s.pkg.Scope().Lookup(name)
	if obj != nil {
		typStr := types2.TypeString(obj.Type(), func(p *types2.Package) string {
			if p == s.pkg {
				return ""
			}
			return s.aliasFor(p.Path())
		})
		return "var " + name + " " + typStr, nil
	}
	return "var " + name + " int", nil
}

func (s *constFnSrc) spliceRange(file *syntax.File, start, end int) (string, error) {
	raw := s.orig[start:end]
	var subs []textEdit
	for _, u := range s.pkgUses {
		if u.file != file {
			continue
		}
		uOff := s.starts[file][u.pos.Line()-1] + int(u.pos.Col()) - 1
		uEnd := uOff + u.len
		if uOff >= start && uEnd <= end {
			alias := s.aliasFor(u.path)
			if alias != "" && alias != string(s.orig[uOff:uEnd]) {
				subs = append(subs, textEdit{
					start: uOff - start,
					end:   uEnd - start,
					text:  alias,
				})
			}
		}
	}
	if len(subs) == 0 {
		return string(raw), nil
	}
	out, err := applyTextEdits(raw, subs)
	if err != nil {
		return "", err
	}
	return string(out), nil
}

func (s *constFnSrc) funcEnd(fd *syntax.FuncDecl) int {
	if fd.Body != nil {
		off := s.off(syntax.EndPos(fd))
		if off < len(s.orig) && s.orig[off] == '}' {
			return off + 1
		}
		return off
	}
	return s.off(syntax.EndPos(fd.Type))
}

func (s *constFnSrc) typeStart(td *syntax.TypeDecl) (int, bool) {
	off := s.off(td.Pos())
	if td.Group != nil {
		specs := s.groupTypeSpecs(s.file, td.Group)
		if len(specs) == 0 {
			return 0, false
		}
		off = s.off(specs[0].Pos())
	}
	return s.keywordStart(token.TYPE, off)
}

func (s *constFnSrc) typeEnd(td *syntax.TypeDecl) int {
	if td.Group != nil {
		specs := s.groupTypeSpecs(s.file, td.Group)
		if len(specs) == 0 {
			return s.off(syntax.EndPos(td))
		}
		last := specs[len(specs)-1]
		return s.scanForwardCloseParen(s.specTypeEnd(last))
	}
	return s.specTypeEnd(td)
}

func (s *constFnSrc) specTypeEnd(td *syntax.TypeDecl) int {
	if e := s.typeExprEnd(td.Type); e > 0 {
		return e
	}
	return s.off(syntax.EndPos(td))
}

func (s *constFnSrc) scanBracedTypeEnd(start int) int {
	depth := 0
	inString := byte(0)
	for off := start; off < len(s.orig); off++ {
		c := s.orig[off]
		if inString != 0 {
			if c == inString && (inString == '`' || s.orig[off-1] != '\\') {
				inString = 0
			}
			continue
		}
		if c == '"' || c == '`' {
			inString = c
			continue
		}
		if c == '\'' {
			off++
			for off < len(s.orig) && s.orig[off] != '\'' {
				if s.orig[off] == '\\' {
					off++
				}
				off++
			}
			continue
		}
		if c == '/' && off+1 < len(s.orig) && s.orig[off+1] == '/' {
			for off < len(s.orig) && s.orig[off] != '\n' {
				off++
			}
			continue
		}
		if c == '/' && off+1 < len(s.orig) && s.orig[off+1] == '*' {
			off += 2
			for off+1 < len(s.orig) && (s.orig[off] != '*' || s.orig[off+1] != '/') {
				off++
			}
			off++
			continue
		}
		if c == '{' {
			depth++
			continue
		}
		if c == '}' {
			depth--
			if depth == 0 {
				return off + 1
			}
			continue
		}
	}
	return start
}

func (s *constFnSrc) constStart(cd *syntax.ConstDecl) (int, bool) {
	off := s.off(cd.Pos())
	if cd.Group != nil {
		specs := s.groupConstSpecs(s.file, cd.Group)
		if len(specs) == 0 {
			return 0, false
		}
		off = s.off(specs[0].Pos())
	}
	return s.keywordStart(token.CONST, off)
}

func (s *constFnSrc) constEnd(cd *syntax.ConstDecl) int {
	if cd.Group != nil {
		specs := s.groupConstSpecs(s.file, cd.Group)
		if len(specs) == 0 {
			return s.off(syntax.EndPos(cd))
		}
		last := specs[len(specs)-1]
		return s.scanForwardCloseParen(s.specConstEnd(last))
	}
	return s.specConstEnd(cd)
}

func (s *constFnSrc) specConstEnd(cd *syntax.ConstDecl) int {
	if cd.Values != nil {
		return s.exprEnd(cd.Values)
	}
	if cd.Type != nil {
		return s.off(syntax.EndPos(cd.Type))
	}
	return s.off(syntax.EndPos(cd))
}

func (s *constFnSrc) groupConstSpecs(file *syntax.File, g *syntax.Group) []*syntax.ConstDecl {
	var specs []*syntax.ConstDecl
	for _, d := range file.DeclList {
		if c, ok := d.(*syntax.ConstDecl); ok && c.Group == g {
			specs = append(specs, c)
		}
	}
	return specs
}

func (s *constFnSrc) groupTypeSpecs(file *syntax.File, g *syntax.Group) []*syntax.TypeDecl {
	var specs []*syntax.TypeDecl
	for _, d := range file.DeclList {
		if t, ok := d.(*syntax.TypeDecl); ok && t.Group == g {
			specs = append(specs, t)
		}
	}
	return specs
}

func (s *constFnSrc) scanForwardCloseParen(afterLast int) int {
	if off := s.skipTriviaFwd(afterLast); off < len(s.orig) && s.orig[off] == ')' {
		return off + 1
	}
	return afterLast
}

func (s *constFnSrc) declEndOfExpr(e syntax.Expr) int {
	if e == nil {
		return 0
	}
	return s.off(syntax.EndPos(e))
}

func (s *constFnSrc) typeExprEnd(e syntax.Expr) int {
	switch e := e.(type) {
	case *syntax.StructType, *syntax.InterfaceType:
		return s.scanBracedTypeEnd(s.off(syntax.StartPos(e)))
	case *syntax.FuncType:
		return s.scanFuncTypeEnd(e)
	case *syntax.MapType:
		return s.typeExprEnd(e.Value)
	case *syntax.ChanType:
		return s.typeExprEnd(e.Elem)
	}
	return s.declEndOfExpr(e)
}

func (s *constFnSrc) scanFuncTypeEnd(ft *syntax.FuncType) int {
	if ft == nil {
		return -1
	}
	off := s.skipTriviaFwd(s.off(syntax.StartPos(ft)) + len("func"))
	if off >= len(s.orig) || s.orig[off] != '(' {
		return -1
	}
	off = s.skipBalanced(off)
	if off < 0 {
		return -1
	}
	off = s.skipTriviaFwd(off)
	if off < len(s.orig) && s.orig[off] == '(' {
		return s.skipBalanced(off)
	}
	if len(ft.ResultList) == 0 {
		return off
	}
	last := ft.ResultList[len(ft.ResultList)-1]
	if last == nil || last.Type == nil {
		return -1
	}
	return s.typeExprEnd(last.Type)
}

func (s *constFnSrc) skipBalanced(start int) int {
	if start >= len(s.orig) {
		return -1
	}
	var close byte
	switch s.orig[start] {
	case '(':
		close = ')'
	case '[':
		close = ']'
	case '{':
		close = '}'
	default:
		return -1
	}
	open := s.orig[start]
	depth := 0
	for off := start; off < len(s.orig); {
		if c := s.orig[off]; c == '"' || c == '`' || c == '\'' {
			off = s.skipString(off)
			continue
		}
		if j := s.skipTriviaFwd(off); j != off {
			off = j
			continue
		}
		if c := s.orig[off]; c == open {
			depth++
		} else if c == close {
			depth--
			if depth == 0 {
				return off + 1
			}
		}
		off++
	}
	return -1
}

func (s *constFnSrc) skipString(off int) int {
	q := s.orig[off]
	if q == '`' {
		for off++; off < len(s.orig) && s.orig[off] != '`'; off++ {
		}
		return off + 1
	}
	off++
	for off < len(s.orig) && s.orig[off] != q {
		if s.orig[off] == '\\' {
			off++
		}
		off++
	}
	return off + 1
}

func (s *constFnSrc) exprEnd(e syntax.Expr) int {
	off := s.off(syntax.EndPos(e))
	if _, ok := e.(*syntax.CallExpr); ok {
		if close, ok := nullaryCallEnd(s.orig, off); ok {
			return close
		}
		if j := s.skipTriviaFwd(off); j < len(s.orig) && s.orig[j] == '(' {
			if k := s.skipTriviaFwd(j + 1); k < len(s.orig) && s.orig[k] == ')' {
				return k + 1
			}
		}
	}
	if j := s.skipTriviaFwd(off); j < len(s.orig) {
		switch s.orig[j] {
		case ')', ']', '}':
			return j + 1
		}
	}
	return off
}

func (s *constFnSrc) skipTriviaFwd(off int) int {
	for off < len(s.orig) {
		c := s.orig[off]
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			off++
			continue
		}
		if c == '/' && off+1 < len(s.orig) {
			if s.orig[off+1] == '/' {
				for off < len(s.orig) && s.orig[off] != '\n' {
					off++
				}
				continue
			}
			if s.orig[off+1] == '*' {
				off += 2
				for off+1 < len(s.orig) && (s.orig[off] != '*' || s.orig[off+1] != '/') {
					off++
				}
				off += 2
				if off > len(s.orig) {
					off = len(s.orig)
				}
				continue
			}
		}
		break
	}
	return off
}

func fileOf(files []*syntax.File, pos syntax.Pos) *syntax.File {
	for _, f := range files {
		if f != nil && f.Pos().Base() == pos.Base() {
			return f
		}
	}
	return nil
}

// KeywordIndex indexes keyword positions in a source file.
type KeywordIndex struct {
	starts map[token.Token][]int
}

func (s *constFnSrc) keywordStart(kind token.Token, off int) (int, bool) {
	idx := s.keywords[s.file]
	if idx == nil {
		idx = new(KeywordIndex)
		s.keywords[s.file] = idx
	}
	return idx.Start(s.orig, kind, off)
}

func (idx *KeywordIndex) scan(src []byte, start int) {
	var s scanner.Scanner
	tf := token.NewFileSet().AddFile("", -1, len(src)-start)
	s.Init(tf, src[start:], nil, 0)
	for {
		pos, tok, _ := s.Scan()
		if tok == token.EOF {
			break
		}
		if tok.IsKeyword() {
			idx.starts[tok] = append(idx.starts[tok], start+tf.Offset(pos))
		}
	}
}

// Append indexes a complete parsed region appended at a newline boundary.
// Before the first lookup, indexing remains lazy.
func (idx *KeywordIndex) Append(src []byte, start int) {
	if idx.starts != nil {
		idx.scan(src, start)
	}
}

// Start returns the last keyword of kind strictly before off in src.
// Calls must use the same source, extended only through Append.
func (idx *KeywordIndex) Start(src []byte, kind token.Token, off int) (int, bool) {
	if idx.starts == nil {
		idx.starts = make(map[token.Token][]int)
		idx.scan(src, 0)
	}
	starts := idx.starts[kind]
	i, _ := slices.BinarySearch(starts, off)
	if i == 0 {
		return -1, false
	}
	return starts[i-1], true
}

func nullaryCallEnd(src []byte, end int) (int, bool) {
	off := end
	for off < len(src) && (src[off] == ' ' || src[off] == '\t' || src[off] == '\n') {
		off++
	}
	if off >= len(src) || src[off] != '(' {
		return 0, false
	}
	off++
	for off < len(src) && (src[off] == ' ' || src[off] == '\t' || src[off] == '\n') {
		off++
	}
	if off >= len(src) || src[off] != ')' {
		return 0, false
	}
	return off + 1, true
}

type constFnImport struct {
	name string
	path string
}

func lineStarts(src []byte) []int {
	starts := []int{0}
	for i, c := range src {
		if c == '\n' {
			starts = append(starts, i+1)
		}
	}
	return starts
}

type textEdit struct {
	start, end int
	text       string
}

func applyTextEdits(src []byte, edits []textEdit) ([]byte, error) {
	slices.SortStableFunc(edits, func(a, b textEdit) int {
		return cmp.Or(cmp.Compare(a.start, b.start), cmp.Compare(a.end, b.end))
	})
	var sb strings.Builder
	prev := 0
	for _, e := range edits {
		if e.start < prev {
			return nil, fmt.Errorf("overlapping edits at byte offset %d", e.start)
		}
		sb.WriteString(string(src[prev:e.start]))
		sb.WriteString(e.text)
		prev = e.end
	}
	sb.WriteString(string(src[prev:]))
	if sb.Len() == 0 {
		return []byte{}, nil
	}
	return []byte(sb.String()), nil
}
