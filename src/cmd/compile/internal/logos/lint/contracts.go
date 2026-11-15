// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"bytes"
	"strings"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

const (
	guardedByRule   = "concurrency::unguarded_access"
	atomicRule      = "concurrency::nonatomic_access"
	lockRequireRule = "concurrency::lock_requirement_unmet"
	layoutRule      = "correctness::layout_constant_mismatch"

	syncPath   = "sync"
	atomicPath = "sync/atomic"
)

// guardPath is a dot-separated path from a receiver or parameter to a lock
// field, as a directive spells it.
type guardPath struct {
	steps []string
}

func (p guardPath) String() string { return strings.Join(p.steps, ".") }

// fieldGuard marks a struct field or package-level variable as protected by
// a lock: Guard runs from the owning struct instance or from package scope
// to the sync.Mutex or sync.RWMutex that guards it.
type fieldGuard struct {
	Object types2.Object
	Guard  guardPath
}

// atomicField marks a struct field or package-level variable as reachable
// only through sync/atomic.
type atomicField struct {
	Object types2.Object
}

// contractEffect is one locks fact of a function.
type contractEffect struct {
	Effect string
	Path   guardPath
}

// funcContract is the lock contract a function declares with
// //logos:requires, //logos:acquires, and //logos:releases.
type funcContract struct {
	Decl    *syntax.FuncDecl
	Object  types2.Object
	Effects []contractEffect
}

// layoutCheck is one //logos:sizeof, //logos:offsetof, or //logos:alignof
// directive: the constant it sits above and the type or field it pins.
type layoutCheck struct {
	Name   *types2.Const
	Kind   string
	Target string
	Pos    syntax.Pos
}

// packageDirectives is the scanned directive set of one package.
type packageDirectives struct {
	Guards    []fieldGuard
	Atomics   []atomicField
	Contracts []funcContract
	Layouts   []layoutCheck
	Misplaced []Result
}

// directiveKinds lists the directive keywords the scanner reads.
var directiveKinds = []string{
	"//logos:guardedby", "//logos:atomic", "//logos:requires",
	"//logos:acquires", "//logos:releases", "//logos:sizeof",
	"//logos:offsetof", "//logos:alignof",
}

func directiveOf(text string) (string, string) {
	for _, k := range directiveKinds {
		if arg, ok := syntax.DirectiveArg(text, k); ok {
			return k, arg
		}
	}
	return "", ""
}

// scanDirectives reads the lock and layout directives of in. Every
// directive must sit on its own line directly above the declaration it
// annotates: a struct field, a package-level variable, a function
// declaration, or a constant. A directive that binds nothing, or that names
// something the rule cannot resolve, is a misplaced result under the rule
// the directive serves.
func scanDirectives(in *Input) *packageDirectives {
	out := &packageDirectives{}
	mode := syntax.Mode(0)
	if in.Kind == KindLgo {
		mode = syntax.LogosMode
	}
	for _, f := range in.Files {
		src, ok := in.Sources[f]
		if !ok || !hasDirective(src) {
			continue
		}
		scanFileDirectives(in, out, directiveFile{file: f, src: src, mode: mode, index: in.buildFileIndex(f)})
	}
	return out
}

func hasDirective(src []byte) bool {
	for _, k := range directiveKinds {
		if bytes.Contains(src, []byte(k)) {
			return true
		}
	}
	return false
}

// directiveFile is one source file prepared for directive scanning.
type directiveFile struct {
	file  *syntax.File
	src   []byte
	mode  syntax.Mode
	index *fileIndex
	lines []string
	base  *syntax.PosBase
}

// fileIndex maps the first line of each annotatable declaration to the
// declaration.
type fileIndex struct {
	fields map[uint]*fieldSite
	funcs  map[uint]*syntax.FuncDecl
	vars   map[uint]*syntax.VarDecl
	consts map[uint]*syntax.ConstDecl
}

// fieldSite is one struct field and the named type that owns it.
type fieldSite struct {
	Field *syntax.Field
	Owner types2.Object
}

func (in *Input) buildFileIndex(f *syntax.File) *fileIndex {
	ix := &fileIndex{
		fields: make(map[uint]*fieldSite),
		funcs:  make(map[uint]*syntax.FuncDecl),
		vars:   make(map[uint]*syntax.VarDecl),
		consts: make(map[uint]*syntax.ConstDecl),
	}
	noteStruct := func(t syntax.Expr, owner types2.Object) {
		st, ok := t.(*syntax.StructType)
		if !ok {
			return
		}
		for _, fld := range st.FieldList {
			ix.fields[fld.Pos().Line()] = &fieldSite{Field: fld, Owner: owner}
		}
	}
	for _, d := range f.DeclList {
		switch d := d.(type) {
		case *syntax.FuncDecl:
			ix.funcs[d.Pos().Line()] = d
		case *syntax.VarDecl:
			ix.vars[d.Pos().Line()] = d
			for _, name := range d.NameList {
				if d.Type != nil {
					noteStruct(d.Type, in.defObject(name))
				}
			}
		case *syntax.ConstDecl:
			ix.consts[d.Pos().Line()] = d
		case *syntax.TypeDecl:
			noteStruct(d.Type, in.defObject(d.Name))
		}
	}
	return ix
}

// directiveSite is one parsed directive comment and where it sits.
type directiveSite struct {
	Kind   string
	Arg    string
	Pos    syntax.Pos
	End    syntax.Pos
	Target uint
}

func scanFileDirectives(in *Input, out *packageDirectives, df directiveFile) {
	df.lines = strings.Split(string(df.src), "\n")
	df.base = df.file.Pos().FileBase()
	syntax.CommentsDoMode(bytes.NewReader(df.src), df.mode, func(line, col uint, text string) {
		kind, arg := directiveOf(strings.TrimRight(text, " \t\r"))
		if kind == "" {
			return
		}
		site := directiveSite{
			Kind:   kind,
			Arg:    arg,
			Pos:    syntax.MakePos(df.base, line, col),
			End:    syntax.MakePos(df.base, line, col+uint(len(text))),
			Target: nextCodeLine(df.lines, line),
		}
		if site.Target == 0 || !ownLine(df.lines, line, col) {
			out.Misplaced = append(out.Misplaced, misplacedDirective(site))
			return
		}
		in.applyDirective(out, df, site)
	})
}

func (in *Input) applyDirective(out *packageDirectives, df directiveFile, site directiveSite) {
	switch site.Kind {
	case "//logos:guardedby":
		in.bindGuard(out, df, site)
	case "//logos:atomic":
		in.bindAtomic(out, df, site)
	case "//logos:requires", "//logos:acquires", "//logos:releases":
		in.bindContract(out, df, site)
	default:
		in.bindLayout(out, df, site)
	}
}

func misplacedDirective(site directiveSite) Result {
	return Result{
		Rule:     ruleOfDirective(site.Kind),
		Severity: Forbid,
		Pos:      site.Pos,
		End:      site.End,
		Message:  site.Kind + " must be on its own line directly above the declaration it annotates",
	}
}

func wrongTarget(site directiveSite, want string) Result {
	return Result{
		Rule:     ruleOfDirective(site.Kind),
		Severity: Forbid,
		Pos:      site.Pos,
		End:      site.End,
		Message:  site.Kind + " must sit above " + want,
	}
}

func ruleOfDirective(kind string) string {
	switch kind {
	case "//logos:requires", "//logos:acquires", "//logos:releases":
		return lockRequireRule
	case "//logos:sizeof", "//logos:offsetof", "//logos:alignof":
		return layoutRule
	case "//logos:atomic":
		return atomicRule
	}
	return guardedByRule
}

func (in *Input) bindGuard(out *packageDirectives, df directiveFile, site directiveSite) {
	path, ok := parseGuardPath(site.Arg)
	if !ok {
		out.Misplaced = append(out.Misplaced, wrongTarget(site, "a mutex field or global"))
		return
	}
	if site2, ok := df.index.fields[site.Target]; ok {
		fieldObj, owner := in.fieldObject(site2)
		if fieldObj == nil || owner == nil {
			out.Misplaced = append(out.Misplaced, wrongTarget(site, "a resolvable struct field"))
			return
		}
		if !in.fieldPathIsMutex(owner, path) {
			out.Misplaced = append(out.Misplaced, wrongTarget(site, "a sync.Mutex or sync.RWMutex field of the same struct"))
			return
		}
		out.Guards = append(out.Guards, fieldGuard{Object: fieldObj, Guard: path})
		return
	}
	decl, ok := df.index.vars[site.Target]
	if !ok {
		out.Misplaced = append(out.Misplaced, wrongTarget(site, "a struct field or a package-level variable"))
		return
	}
	ov := in.globalGuardObject(decl, path, site)
	if ov.misplaced != nil {
		out.Misplaced = append(out.Misplaced, *ov.misplaced)
		return
	}
	out.Guards = append(out.Guards, fieldGuard{Object: ov.object, Guard: path})
}

// guardObject is the result of resolving a package-level guard.
type guardObject struct {
	object    types2.Object
	misplaced *Result
}

func (in *Input) globalGuardObject(decl *syntax.VarDecl, path guardPath, site directiveSite) guardObject {
	root := in.Pkg.Scope().Lookup(path.steps[0])
	if root == nil || !in.pathIsMutexFrom(root.Type(), path.steps[1:]) {
		r := wrongTarget(site, "a sync.Mutex or sync.RWMutex global")
		return guardObject{misplaced: &r}
	}
	for _, name := range decl.NameList {
		if obj := in.defObject(name); obj != nil {
			return guardObject{object: obj}
		}
	}
	r := wrongTarget(site, "a package-level variable")
	return guardObject{misplaced: &r}
}

func (in *Input) bindAtomic(out *packageDirectives, df directiveFile, site directiveSite) {
	if site.Arg != "" {
		out.Misplaced = append(out.Misplaced, wrongTarget(site, "a field or global and no argument"))
		return
	}
	if site2, ok := df.index.fields[site.Target]; ok {
		if obj, _ := in.fieldObject(site2); obj != nil {
			out.Atomics = append(out.Atomics, atomicField{Object: obj})
			return
		}
		out.Misplaced = append(out.Misplaced, wrongTarget(site, "a resolvable struct field"))
		return
	}
	decl, ok := df.index.vars[site.Target]
	if !ok {
		out.Misplaced = append(out.Misplaced, wrongTarget(site, "a struct field or a package-level variable"))
		return
	}
	for _, name := range decl.NameList {
		if obj := in.defObject(name); obj != nil {
			out.Atomics = append(out.Atomics, atomicField{Object: obj})
			return
		}
	}
	out.Misplaced = append(out.Misplaced, wrongTarget(site, "a package-level variable"))
}

func (in *Input) bindContract(out *packageDirectives, df directiveFile, site directiveSite) {
	effect := strings.TrimPrefix(site.Kind, "//logos:")
	path, ok := parseGuardPath(site.Arg)
	if !ok {
		out.Misplaced = append(out.Misplaced, wrongTarget(site, "a function and a lock path"))
		return
	}
	decl, ok := df.index.funcs[site.Target]
	if !ok {
		out.Misplaced = append(out.Misplaced, wrongTarget(site, "a function declaration"))
		return
	}
	obj := in.defObject(decl.Name)
	if obj == nil {
		out.Misplaced = append(out.Misplaced, wrongTarget(site, "a resolvable function"))
		return
	}
	if !in.contractPathResolves(decl, path) {
		out.Misplaced = append(out.Misplaced, wrongTarget(site, "a sync.Mutex or sync.RWMutex reached from a parameter or the receiver"))
		return
	}
	for i := range out.Contracts {
		if out.Contracts[i].Decl == decl {
			out.Contracts[i].Effects = append(out.Contracts[i].Effects, contractEffect{Effect: effect, Path: path})
			return
		}
	}
	out.Contracts = append(out.Contracts, funcContract{
		Decl:    decl,
		Object:  obj,
		Effects: []contractEffect{{Effect: effect, Path: path}},
	})
}

func (in *Input) bindLayout(out *packageDirectives, df directiveFile, site directiveSite) {
	name := strings.TrimPrefix(site.Kind, "//logos:")
	if site.Arg == "" || strings.Contains(site.Arg, " ") {
		out.Misplaced = append(out.Misplaced, wrongTarget(site, "a constant and a type name"))
		return
	}
	decl, ok := df.index.consts[site.Target]
	if !ok {
		if vd, isVar := df.index.vars[site.Target]; isVar && len(vd.NameList) > 0 {
			res := wrongTarget(site, "a constant declaration")
			res.Pos = vd.NameList[0].Pos()
			res.End = res.Pos
			out.Misplaced = append(out.Misplaced, res)
			return
		}
		out.Misplaced = append(out.Misplaced, wrongTarget(site, "a constant declaration"))
		return
	}
	if len(decl.NameList) != 1 {
		out.Misplaced = append(out.Misplaced, wrongTarget(site, "a constant declaration with one name"))
		return
	}
	obj, ok := in.defObject(decl.NameList[0]).(*types2.Const)
	if !ok {
		out.Misplaced = append(out.Misplaced, wrongTarget(site, "a constant declaration"))
		return
	}
	out.Layouts = append(out.Layouts, layoutCheck{Name: obj, Kind: name, Target: site.Arg, Pos: site.Pos})
}

func parseGuardPath(arg string) (guardPath, bool) {
	if arg == "" || strings.ContainsAny(arg, " \t") {
		return guardPath{}, false
	}
	steps := strings.Split(arg, ".")
	for _, s := range steps {
		if !isDirectiveIdent(s) {
			return guardPath{}, false
		}
	}
	return guardPath{steps: steps}, true
}

func isDirectiveIdent(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		switch {
		case r == '_' || 'a' <= r && r <= 'z' || 'A' <= r && r <= 'Z':
		case i > 0 && '0' <= r && r <= '9':
		default:
			return false
		}
	}
	return true
}

func (in *Input) defObject(name *syntax.Name) types2.Object {
	if in.Info == nil || in.Info.Defs == nil {
		return nil
	}
	return in.Info.Defs[name]
}

func (in *Input) useObject(name *syntax.Name) types2.Object {
	if in.Info == nil || in.Info.Uses == nil {
		return nil
	}
	if obj := in.Info.Uses[name]; obj != nil {
		return obj
	}
	return in.defObject(name)
}

func (in *Input) fieldObject(site *fieldSite) (types2.Object, *types2.Named) {
	if site.Field.Name == nil {
		return nil, nil
	}
	obj := in.defObject(site.Field.Name)
	if obj == nil {
		obj = in.useObject(site.Field.Name)
	}
	if obj == nil {
		return nil, nil
	}
	owner, ok := types2.Unalias(site.Owner.Type()).(*types2.Named)
	if !ok {
		return nil, nil
	}
	if _, ok := owner.Underlying().(*types2.Struct); !ok {
		return nil, nil
	}
	return obj, owner
}

func ownerNamed(obj types2.Object) *types2.Named {
	if obj == nil {
		return nil
	}
	named, ok := types2.Unalias(obj.Type()).(*types2.Named)
	if !ok {
		return nil
	}
	if _, ok := named.Underlying().(*types2.Struct); !ok {
		return nil
	}
	return named
}

// fieldPathIsMutex reports whether named reaches a mutex through path.
func (in *Input) fieldPathIsMutex(named *types2.Named, path guardPath) bool {
	return in.pathIsMutexFrom(named, path.steps)
}

// pathIsMutexFrom reports whether walking the field names of steps from t
// ends at a mutex.
func (in *Input) pathIsMutexFrom(t types2.Type, steps []string) bool {
	for _, step := range steps {
		st, ok := structOf(t)
		if !ok {
			return false
		}
		field := fieldNamed(st, step)
		if field == nil {
			return false
		}
		t = field.Type()
	}
	return isMutexType(t)
}

// structOf returns the underlying struct of t, looking through named types
// and one pointer.
func structOf(t types2.Type) (*types2.Struct, bool) {
	if ptr, ok := types2.Unalias(t).(*types2.Pointer); ok {
		t = ptr.Elem()
	}
	if named, ok := types2.Unalias(t).(*types2.Named); ok {
		t = named.Underlying()
	}
	st, ok := types2.Unalias(t).(*types2.Struct)
	return st, ok
}

// fieldNamed returns the field of st with the given name, or nil.
func fieldNamed(st *types2.Struct, name string) types2.Object {
	for i := range st.NumFields() {
		if st.Field(i).Name() == name {
			return st.Field(i)
		}
	}
	return nil
}

func (in *Input) contractPathResolves(decl *syntax.FuncDecl, path guardPath) bool {
	obj := in.declParamObject(decl, path.steps[0])
	if obj == nil {
		return false
	}
	if len(path.steps) == 1 {
		return isMutexType(obj.Type())
	}
	return in.pathIsMutexFrom(obj.Type(), path.steps[1:])
}

func (in *Input) declParamObject(decl *syntax.FuncDecl, name string) types2.Object {
	if decl.Recv != nil {
		if recvName := decl.Recv.Name; recvName != nil && recvName.Value == name {
			if obj := in.defObject(recvName); obj != nil {
				return obj
			}
		}
	}
	for _, field := range decl.Type.ParamList {
		if field.Name != nil && field.Name.Value == name {
			if obj := in.defObject(field.Name); obj != nil {
				return obj
			}
		}
	}
	return nil
}

func isMutexType(t types2.Type) bool {
	if obj := namedTypeObject(t); obj != nil {
		return obj.Pkg() != nil && obj.Pkg().Path() == syncPath && (obj.Name() == "Mutex" || obj.Name() == "RWMutex")
	}
	return false
}

func isAtomicType(t types2.Type) bool {
	obj := namedTypeObject(t)
	if obj == nil || obj.Pkg() == nil || obj.Pkg().Path() != atomicPath {
		return false
	}
	switch obj.Name() {
	case "Bool", "Int32", "Int64", "Uint32", "Uint64", "Uintptr", "Pointer", "Value":
		return true
	}
	return false
}

// namedTypeObject returns the type object of t after one pointer and an
// alias peel, or nil.
func namedTypeObject(t types2.Type) *types2.TypeName {
	if ptr, ok := types2.Unalias(t).(*types2.Pointer); ok {
		t = ptr.Elem()
	}
	named, ok := types2.Unalias(t).(*types2.Named)
	if !ok {
		return nil
	}
	return named.Obj()
}
