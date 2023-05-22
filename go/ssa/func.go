// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file implements the Function type.

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"io"
	"os"
	"strings"

	"golang.org/x/tools/internal/typeparams"
)

// Like ObjectOf, but panics instead of returning nil.
// Only valid during f's create and build phases.
func (f *Function) objectOf(id *ast.Ident) types.Object {
	if o := f.info.ObjectOf(id); o != nil {
		return o
	}
	panic(fmt.Sprintf("no types.Object for ast.Ident %s @ %s",
		id.Name, f.Prog.Fset.Position(id.Pos())))
}

// Like TypeOf, but panics instead of returning nil.
// Only valid during f's create and build phases.
func (f *Function) typeOf(e ast.Expr) types.Type {
	if T := f.info.TypeOf(e); T != nil {
		return f.typ(T)
	}
	panic(fmt.Sprintf("no type for %T @ %s", e, f.Prog.Fset.Position(e.Pos())))
}

// typ is the locally instantiated type of T. T==typ(T) if f is not an instantiation.
func (f *Function) typ(T types.Type) types.Type {
	return f.subst.typ(T)
}

// If id is an Instance, returns info.Instances[id].Type.
// Otherwise returns f.typeOf(id).
func (f *Function) instanceType(id *ast.Ident) types.Type {
	if t, ok := typeparams.GetInstances(f.info)[id]; ok {
		return t.Type
	}
	return f.typeOf(id)
}

// selection returns a *selection corresponding to f.info.Selections[selector]
// with potential updates for type substitution.
func (f *Function) selection(selector *ast.SelectorExpr) *selection {
	sel := f.info.Selections[selector]
	if sel == nil {
		return nil
	}

	switch sel.Kind() {
	case types.MethodExpr, types.MethodVal:
		if recv := f.typ(sel.Recv()); recv != sel.Recv() {
			// recv changed during type substitution.
			pkg := f.declaredPackage().Pkg
			obj, index, indirect := types.LookupFieldOrMethod(recv, true, pkg, sel.Obj().Name())

			// sig replaces sel.Type(). See (types.Selection).Typ() for details.
			sig := obj.Type().(*types.Signature)
			sig = changeRecv(sig, newVar(sig.Recv().Name(), recv))
			if sel.Kind() == types.MethodExpr {
				sig = recvAsFirstArg(sig)
			}
			return &selection{
				kind:     sel.Kind(),
				recv:     recv,
				typ:      sig,
				obj:      obj,
				index:    index,
				indirect: indirect,
			}
		}
	}
	return toSelection(sel)
}

// Destinations associated with unlabelled for/switch/select stmts.
// We push/pop one of these as we enter/leave each construct and for
// each BranchStmt we scan for the innermost target of the right type.
type targets struct {
	tail         *targets // rest of stack
	_break       *BasicBlock
	_continue    *BasicBlock
	_fallthrough *BasicBlock
}

// Destinations associated with a labelled block.
// We populate these as labels are encountered in forward gotos or
// labelled statements.
type lblock struct {
	_goto     *BasicBlock
	_break    *BasicBlock
	_continue *BasicBlock
}

// labelledBlock returns the branch target associated with the
// specified label, creating it if needed.
func (f *Function) labelledBlock(label *ast.Ident) *lblock {
	obj := f.objectOf(label)
	lb := f.lblocks[obj]
	if lb == nil {
		lb = &lblock{_goto: f.newBasicBlock(label.Name)}
		if f.lblocks == nil {
			f.lblocks = make(map[types.Object]*lblock)
		}
		f.lblocks[obj] = lb
	}
	return lb
}

// addParam adds a (non-escaping) parameter to f.Params of the
// specified name, type and source position.
func (f *Function) addParam(name string, typ types.Type, pos token.Pos) *Parameter {
	v := &Parameter{
		name:   name,
		typ:    typ,
		pos:    pos,
		parent: f,
	}
	f.Params = append(f.Params, v)
	return v
}

func (f *Function) addParamObj(obj types.Object) *Parameter {
	name := obj.Name()
	if name == "" {
		name = fmt.Sprintf("arg%d", len(f.Params))
	}
	param := f.addParam(name, f.typ(obj.Type()), obj.Pos())
	param.object = obj
	return param
}

// addSpilledParam declares a parameter that is pre-spilled to the
// stack; the function body will load/store the spilled location.
// Subsequent lifting will eliminate spills where possible.
func (f *Function) addSpilledParam(obj types.Object) {
	param := f.addParamObj(obj)
	spill := &Alloc{Comment: obj.Name()}
	spill.setType(types.NewPointer(param.Type()))
	spill.setPos(obj.Pos())
	f.objects[obj] = spill
	f.Locals = append(f.Locals, spill)
	f.emit(spill)
	f.emit(&Store{Addr: spill, Val: param})
}

// startBody initializes the function prior to generating SSA code for its body.
// Precondition: f.Type() already set.
func (f *Function) startBody() {
	f.currentBlock = f.newBasicBlock("entry")
	f.objects = make(map[types.Object]Value) // needed for some synthetics, e.g. init
}

// createSyntacticParams populates f.Params and generates code (spills
// and named result locals) for all the parameters declared in the
// syntax.  In addition it populates the f.objects mapping.
//
// Preconditions:
// f.startBody() was called. f.info != nil.
// Postcondition:
// len(f.Params) == len(f.Signature.Params) + (f.Signature.Recv() ? 1 : 0)
func (f *Function) createSyntacticParams(recv *ast.FieldList, functype *ast.FuncType) {
	// Receiver (at most one inner iteration).
	if recv != nil {
		for _, field := range recv.List {
			for _, n := range field.Names {
				f.addSpilledParam(f.info.Defs[n])
			}
			// Anonymous receiver?  No need to spill.
			if field.Names == nil {
				f.addParamObj(f.Signature.Recv())
			}
		}
	}

	// Parameters.
	if functype.Params != nil {
		n := len(f.Params) // 1 if has recv, 0 otherwise
		for _, field := range functype.Params.List {
			for _, n := range field.Names {
				f.addSpilledParam(f.info.Defs[n])
			}
			// Anonymous parameter?  No need to spill.
			if field.Names == nil {
				f.addParamObj(f.Signature.Params().At(len(f.Params) - n))
			}
		}
	}

	// Named results.
	if functype.Results != nil {
		for _, field := range functype.Results.List {
			// Implicit "var" decl of locals for named results.
			for _, n := range field.Names {
				f.namedResults = append(f.namedResults, f.addLocalForIdent(n))
			}
		}
	}
}

type setNumable interface {
	setNum(int)
}

// numberRegisters assigns numbers to all SSA registers
// (value-defining Instructions) in f, to aid debugging.
// (Non-Instruction Values are named at construction.)
func numberRegisters(f *Function) {
	v := 0
	for _, b := range f.Blocks {
		for _, instr := range b.Instrs {
			switch instr.(type) {
			case Value:
				instr.(setNumable).setNum(v)
				v++
			}
		}
	}
}

// buildReferrers populates the def/use information in all non-nil
// Value.Referrers slice.
// Precondition: all such slices are initially empty.
func buildReferrers(f *Function) {
	var rands []*Value
	for _, b := range f.Blocks {
		for _, instr := range b.Instrs {
			rands = instr.Operands(rands[:0]) // recycle storage
			for _, rand := range rands {
				if r := *rand; r != nil {
					if ref := r.Referrers(); ref != nil {
						*ref = append(*ref, instr)
					}
				}
			}
		}
	}
}

// mayNeedRuntimeTypes returns all of the types in the body of fn that might need runtime types.
//
// EXCLUSIVE_LOCKS_ACQUIRED(meth.Prog.methodsMu)
func mayNeedRuntimeTypes(fn *Function) []types.Type {
	// Collect all types that may need rtypes, i.e. those that flow into an interface.
	var ts []types.Type
	for _, bb := range fn.Blocks {
		for _, instr := range bb.Instrs {
			if mi, ok := instr.(*MakeInterface); ok {
				ts = append(ts, mi.X.Type())
			}
		}
	}

	// Types that contain a parameterized type are considered to not be runtime types.
	if fn.typeparams.Len() == 0 {
		return ts // No potentially parameterized types.
	}
	// Filter parameterized types, in place.
	fn.Prog.methodsMu.Lock()
	defer fn.Prog.methodsMu.Unlock()
	filtered := ts[:0]
	for _, t := range ts {
		if !fn.Prog.parameterized.isParameterized(t) {
			filtered = append(filtered, t)
		}
	}
	return filtered
}

// finishBody() finalizes the contents of the function after SSA code generation of its body.
//
// The function is not done being built until done() is called.
func (f *Function) finishBody() {
	f.objects = nil
	f.currentBlock = nil
	f.lblocks = nil

	// Don't pin the AST in memory (except in debug mode).
	if n := f.syntax; n != nil && !f.debugInfo() {
		f.syntax = extentNode{n.Pos(), n.End()}
	}

	// Remove from f.Locals any Allocs that escape to the heap.
	j := 0
	for _, l := range f.Locals {
		if !l.Heap {
			f.Locals[j] = l
			j++
		}
	}
	// Nil out f.Locals[j:] to aid GC.
	for i := j; i < len(f.Locals); i++ {
		f.Locals[i] = nil
	}
	f.Locals = f.Locals[:j]

	optimizeBlocks(f)

	buildReferrers(f)

	buildDomTree(f)

	if f.Prog.mode&NaiveForm == 0 {
		// For debugging pre-state of lifting pass:
		// numberRegisters(f)
		// f.WriteTo(os.Stderr)
		lift(f)
	}

	// clear remaining stateful variables
	f.namedResults = nil // (used by lifting)
	f.info = nil
	f.subst = nil

	numberRegisters(f) // uses f.namedRegisters
}

// After this, function is done with BUILD phase.
func (f *Function) done() {
	assert(f.parent == nil, "done called on an anonymous function")

	var visit func(*Function)
	visit = func(f *Function) {
		for _, anon := range f.AnonFuncs {
			visit(anon) // anon is done building before f.
		}

		f.built = true // function is done with BUILD phase

		if f.Prog.mode&PrintFunctions != 0 {
			printMu.Lock()
			f.WriteTo(os.Stdout)
			printMu.Unlock()
		}

		if f.Prog.mode&SanityCheckFunctions != 0 {
			mustSanityCheck(f, nil)
		}
	}
	visit(f)
}

// removeNilBlocks eliminates nils from f.Blocks and updates each
// BasicBlock.Index.  Use this after any pass that may delete blocks.
func (f *Function) removeNilBlocks() {
	j := 0
	for _, b := range f.Blocks {
		if b != nil {
			b.Index = j
			f.Blocks[j] = b
			j++
		}
	}
	// Nil out f.Blocks[j:] to aid GC.
	for i := j; i < len(f.Blocks); i++ {
		f.Blocks[i] = nil
	}
	f.Blocks = f.Blocks[:j]
}

// SetDebugMode sets the debug mode for package pkg.  If true, all its
// functions will include full debug info.  This greatly increases the
// size of the instruction stream, and causes Functions to depend upon
// the ASTs, potentially keeping them live in memory for longer.
func (pkg *Package) SetDebugMode(debug bool) {
	// TODO(adonovan): do we want ast.File granularity?
	pkg.debug = debug
}

// debugInfo reports whether debug info is wanted for this function.
func (f *Function) debugInfo() bool {
	// debug info for instantiations follows the debug info of their origin.
	p := f.declaredPackage()
	return p != nil && p.debug
}

// addNamedLocal creates a local variable, adds it to function f and
// returns it.  Its name and type are taken from obj.  Subsequent
// calls to f.lookup(obj) will return the same local.
func (f *Function) addNamedLocal(obj types.Object) *Alloc {
	l := f.addLocal(obj.Type(), obj.Pos())
	l.Comment = obj.Name()
	f.objects[obj] = l
	return l
}

func (f *Function) addLocalForIdent(id *ast.Ident) *Alloc {
	return f.addNamedLocal(f.info.Defs[id])
}

// addLocal creates an anonymous local variable of type typ, adds it
// to function f and returns it.  pos is the optional source location.
func (f *Function) addLocal(typ types.Type, pos token.Pos) *Alloc {
	typ = f.typ(typ)
	v := &Alloc{}
	v.setType(types.NewPointer(typ))
	v.setPos(pos)
	f.Locals = append(f.Locals, v)
	f.emit(v)
	return v
}

// lookup returns the address of the named variable identified by obj
// that is local to function f or one of its enclosing functions.
// If escaping, the reference comes from a potentially escaping pointer
// expression and the referent must be heap-allocated.
func (f *Function) lookup(obj types.Object, escaping bool) Value {
	if v, ok := f.objects[obj]; ok {
		if alloc, ok := v.(*Alloc); ok && escaping {
			alloc.Heap = true
		}
		return v // function-local var (address)
	}

	// Definition must be in an enclosing function;
	// plumb it through intervening closures.
	if f.parent == nil {
		panic("no ssa.Value for " + obj.String())
	}
	outer := f.parent.lookup(obj, true) // escaping
	v := &FreeVar{
		name:   obj.Name(),
		typ:    outer.Type(),
		pos:    outer.Pos(),
		outer:  outer,
		parent: f,
	}
	f.objects[obj] = v
	f.FreeVars = append(f.FreeVars, v)
	return v
}

// emit emits the specified instruction to function f.
func (f *Function) emit(instr Instruction) Value {
	return f.currentBlock.emit(instr)
}

// RelString returns the full name of this function, qualified by
// package name, receiver type, etc.
//
// The specific formatting rules are not guaranteed and may change.
//
// Examples:
//
//	"math.IsNaN"                  // a package-level function
//	"(*bytes.Buffer).Bytes"       // a declared method or a wrapper
//	"(*bytes.Buffer).Bytes$thunk" // thunk (func wrapping method; receiver is param 0)
//	"(*bytes.Buffer).Bytes$bound" // bound (func wrapping method; receiver supplied by closure)
//	"main.main$1"                 // an anonymous function in main
//	"main.init#1"                 // a declared init function
//	"main.init"                   // the synthesized package initializer
//
// When these functions are referred to from within the same package
// (i.e. from == f.Pkg.Object), they are rendered without the package path.
// For example: "IsNaN", "(*Buffer).Bytes", etc.
//
// All non-synthetic functions have distinct package-qualified names.
// (But two methods may have the same name "(T).f" if one is a synthetic
// wrapper promoting a non-exported method "f" from another package; in
// that case, the strings are equal but the identifiers "f" are distinct.)
func (f *Function) RelString(from *types.Package) string {
	// Anonymous?
	if f.parent != nil {
		// An anonymous function's Name() looks like "parentName$1",
		// but its String() should include the type/package/etc.
		parent := f.parent.RelString(from)
		for i, anon := range f.parent.AnonFuncs {
			if anon == f {
				return fmt.Sprintf("%s$%d", parent, 1+i)
			}
		}

		return f.name // should never happen
	}

	// Method (declared or wrapper)?
	if recv := f.Signature.Recv(); recv != nil {
		return f.relMethod(from, recv.Type())
	}

	// Thunk?
	if f.method != nil {
		return f.relMethod(from, f.method.recv)
	}

	// Bound?
	if len(f.FreeVars) == 1 && strings.HasSuffix(f.name, "$bound") {
		return f.relMethod(from, f.FreeVars[0].Type())
	}

	// Package-level function?
	// Prefix with package name for cross-package references only.
	if p := f.relPkg(); p != nil && p != from {
		return fmt.Sprintf("%s.%s", p.Path(), f.name)
	}

	// Unknown.
	return f.name
}

func (f *Function) relMethod(from *types.Package, recv types.Type) string {
	return fmt.Sprintf("(%s).%s", relType(recv, from), f.name)
}

// writeSignature writes to buf the signature sig in declaration syntax.
func writeSignature(buf *bytes.Buffer, from *types.Package, name string, sig *types.Signature) {
	buf.WriteString("func ")
	if recv := sig.Recv(); recv != nil {
		buf.WriteString("(")
		if name := recv.Name(); name != "" {
			buf.WriteString(name)
			buf.WriteString(" ")
		}
		types.WriteType(buf, recv.Type(), types.RelativeTo(from))
		buf.WriteString(") ")
	}
	buf.WriteString(name)
	types.WriteSignature(buf, sig, types.RelativeTo(from))
}

// declaredPackage returns the package fn is declared in or nil if the
// function is not declared in a package.
func (fn *Function) declaredPackage() *Package {
	switch {
	case fn.Pkg != nil:
		return fn.Pkg // non-generic function
	case fn.topLevelOrigin != nil:
		return fn.topLevelOrigin.Pkg // instance of a named generic function
	case fn.parent != nil:
		return fn.parent.declaredPackage() // instance of an anonymous [generic] function
	default:
		return nil // function is not declared in a package, e.g. a wrapper.
	}
}

// relPkg returns types.Package fn is printed in relationship to.
func (fn *Function) relPkg() *types.Package {
	if p := fn.declaredPackage(); p != nil {
		return p.Pkg
	}
	return nil
}

var _ io.WriterTo = (*Function)(nil) // *Function implements io.Writer

func (f *Function) WriteTo(w io.Writer) (int64, error) {
	var buf bytes.Buffer
	WriteFunction(&buf, f)
	n, err := w.Write(buf.Bytes())
	return int64(n), err
}

// WriteFunction writes to buf a human-readable "disassembly" of f.
func WriteFunction(buf *bytes.Buffer, f *Function) {
	fmt.Fprintf(buf, "# Name: %s\n", f.String())
	if f.Pkg != nil {
		fmt.Fprintf(buf, "# Package: %s\n", f.Pkg.Pkg.Path())
	}
	if syn := f.Synthetic; syn != "" {
		fmt.Fprintln(buf, "# Synthetic:", syn)
	}
	if pos := f.Pos(); pos.IsValid() {
		fmt.Fprintf(buf, "# Location: %s\n", f.Prog.Fset.Position(pos))
	}

	if f.parent != nil {
		fmt.Fprintf(buf, "# Parent: %s\n", f.parent.Name())
	}

	if f.Recover != nil {
		fmt.Fprintf(buf, "# Recover: %s\n", f.Recover)
	}

	from := f.relPkg()

	if f.FreeVars != nil {
		buf.WriteString("# Free variables:\n")
		for i, fv := range f.FreeVars {
			fmt.Fprintf(buf, "# % 3d:\t%s %s\n", i, fv.Name(), relType(fv.Type(), from))
		}
	}

	if len(f.Locals) > 0 {
		buf.WriteString("# Locals:\n")
		for i, l := range f.Locals {
			fmt.Fprintf(buf, "# % 3d:\t%s %s\n", i, l.Name(), relType(mustDeref(l.Type()), from))
		}
	}
	writeSignature(buf, from, f.Name(), f.Signature)
	buf.WriteString(":\n")

	if f.Blocks == nil {
		buf.WriteString("\t(external)\n")
	}

	// NB. column calculations are confused by non-ASCII
	// characters and assume 8-space tabs.
	const punchcard = 80 // for old time's sake.
	const tabwidth = 8
	for _, b := range f.Blocks {
		if b == nil {
			// Corrupt CFG.
			fmt.Fprintf(buf, ".nil:\n")
			continue
		}
		n, _ := fmt.Fprintf(buf, "%d:", b.Index)
		bmsg := fmt.Sprintf("%s P:%d S:%d", b.Comment, len(b.Preds), len(b.Succs))
		fmt.Fprintf(buf, "%*s%s\n", punchcard-1-n-len(bmsg), "", bmsg)

		if false { // CFG debugging
			fmt.Fprintf(buf, "\t# CFG: %s --> %s --> %s\n", b.Preds, b, b.Succs)
		}
		for _, instr := range b.Instrs {
			buf.WriteString("\t")
			switch v := instr.(type) {
			case Value:
				l := punchcard - tabwidth
				// Left-align the instruction.
				if name := v.Name(); name != "" {
					n, _ := fmt.Fprintf(buf, "%s = ", name)
					l -= n
				}
				n, _ := buf.WriteString(instr.String())
				l -= n
				// Right-align the type if there's space.
				if t := v.Type(); t != nil {
					buf.WriteByte(' ')
					ts := relType(t, from)
					l -= len(ts) + len("  ") // (spaces before and after type)
					if l > 0 {
						fmt.Fprintf(buf, "%*s", l, "")
					}
					buf.WriteString(ts)
				}
			case nil:
				// Be robust against bad transforms.
				buf.WriteString("<deleted>")
			default:
				buf.WriteString(instr.String())
			}
			buf.WriteString("\n")
		}
	}
	fmt.Fprintf(buf, "\n")
}

// newBasicBlock adds to f a new basic block and returns it.  It does
// not automatically become the current block for subsequent calls to emit.
// comment is an optional string for more readable debugging output.
func (f *Function) newBasicBlock(comment string) *BasicBlock {
	b := &BasicBlock{
		Index:   len(f.Blocks),
		Comment: comment,
		parent:  f,
	}
	b.Succs = b.succs2[:0]
	f.Blocks = append(f.Blocks, b)
	return b
}

// NewFunction returns a new synthetic Function instance belonging to
// prog, with its name and signature fields set as specified.
//
// The caller is responsible for initializing the remaining fields of
// the function object, e.g. Pkg, Params, Blocks.
//
// It is practically impossible for clients to construct well-formed
// SSA functions/packages/programs directly, so we assume this is the
// job of the Builder alone.  NewFunction exists to provide clients a
// little flexibility.  For example, analysis tools may wish to
// construct fake Functions for the root of the callgraph, a fake
// "reflect" package, etc.
//
// TODO(adonovan): think harder about the API here.
func (prog *Program) NewFunction(name string, sig *types.Signature, provenance string) *Function {
	return &Function{Prog: prog, name: name, Signature: sig, Synthetic: provenance}
}

type extentNode [2]token.Pos

func (n extentNode) Pos() token.Pos { return n[0] }
func (n extentNode) End() token.Pos { return n[1] }

// Syntax returns an ast.Node whose Pos/End methods provide the
// lexical extent of the function if it was defined by Go source code
// (f.Synthetic==""), or nil otherwise.
//
// If f was built with debug information (see Package.SetDebugRef),
// the result is the *ast.FuncDecl or *ast.FuncLit that declared the
// function.  Otherwise, it is an opaque Node providing only position
// information; this avoids pinning the AST in memory.
func (f *Function) Syntax() ast.Node { return f.syntax }
