// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file implements the Function and BasicBlock types.

import (
	"fmt"
	"go/ast"
	"go/token"
	"io"
	"os"
	"strings"

	"code.google.com/p/go.tools/go/types"
)

// addEdge adds a control-flow graph edge from from to to.
func addEdge(from, to *BasicBlock) {
	from.Succs = append(from.Succs, to)
	to.Preds = append(to.Preds, from)
}

// Parent returns the function that contains block b.
func (b *BasicBlock) Parent() *Function { return b.parent }

// String returns a human-readable label of this block.
// It is not guaranteed unique within the function.
//
func (b *BasicBlock) String() string {
	return fmt.Sprintf("%d.%s", b.Index, b.Comment)
}

// emit appends an instruction to the current basic block.
// If the instruction defines a Value, it is returned.
//
func (b *BasicBlock) emit(i Instruction) Value {
	i.setBlock(b)
	b.Instrs = append(b.Instrs, i)
	v, _ := i.(Value)
	return v
}

// predIndex returns the i such that b.Preds[i] == c or panics if
// there is none.
func (b *BasicBlock) predIndex(c *BasicBlock) int {
	for i, pred := range b.Preds {
		if pred == c {
			return i
		}
	}
	panic(fmt.Sprintf("no edge %s -> %s", c, b))
}

// hasPhi returns true if b.Instrs contains φ-nodes.
func (b *BasicBlock) hasPhi() bool {
	_, ok := b.Instrs[0].(*Phi)
	return ok
}

// phis returns the prefix of b.Instrs containing all the block's φ-nodes.
func (b *BasicBlock) phis() []Instruction {
	for i, instr := range b.Instrs {
		if _, ok := instr.(*Phi); !ok {
			return b.Instrs[:i]
		}
	}
	return nil // unreachable in well-formed blocks
}

// replacePred replaces all occurrences of p in b's predecessor list with q.
// Ordinarily there should be at most one.
//
func (b *BasicBlock) replacePred(p, q *BasicBlock) {
	for i, pred := range b.Preds {
		if pred == p {
			b.Preds[i] = q
		}
	}
}

// replaceSucc replaces all occurrences of p in b's successor list with q.
// Ordinarily there should be at most one.
//
func (b *BasicBlock) replaceSucc(p, q *BasicBlock) {
	for i, succ := range b.Succs {
		if succ == p {
			b.Succs[i] = q
		}
	}
}

// removePred removes all occurrences of p in b's
// predecessor list and φ-nodes.
// Ordinarily there should be at most one.
//
func (b *BasicBlock) removePred(p *BasicBlock) {
	phis := b.phis()

	// We must preserve edge order for φ-nodes.
	j := 0
	for i, pred := range b.Preds {
		if pred != p {
			b.Preds[j] = b.Preds[i]
			// Strike out φ-edge too.
			for _, instr := range phis {
				phi := instr.(*Phi)
				phi.Edges[j] = phi.Edges[i]
			}
			j++
		}
	}
	// Nil out b.Preds[j:] and φ-edges[j:] to aid GC.
	for i := j; i < len(b.Preds); i++ {
		b.Preds[i] = nil
		for _, instr := range phis {
			instr.(*Phi).Edges[i] = nil
		}
	}
	b.Preds = b.Preds[:j]
	for _, instr := range phis {
		phi := instr.(*Phi)
		phi.Edges = phi.Edges[:j]
	}
}

// Destinations associated with unlabelled for/switch/select stmts.
// We push/pop one of these as we enter/leave each construct and for
// each BranchStmt we scan for the innermost target of the right type.
//
type targets struct {
	tail         *targets // rest of stack
	_break       *BasicBlock
	_continue    *BasicBlock
	_fallthrough *BasicBlock
}

// Destinations associated with a labelled block.
// We populate these as labels are encountered in forward gotos or
// labelled statements.
//
type lblock struct {
	_goto     *BasicBlock
	_break    *BasicBlock
	_continue *BasicBlock
}

// labelledBlock returns the branch target associated with the
// specified label, creating it if needed.
//
func (f *Function) labelledBlock(label *ast.Ident) *lblock {
	lb := f.lblocks[label.Obj]
	if lb == nil {
		lb = &lblock{_goto: f.newBasicBlock(label.Name)}
		if f.lblocks == nil {
			f.lblocks = make(map[*ast.Object]*lblock)
		}
		f.lblocks[label.Obj] = lb
	}
	return lb
}

// addParam adds a (non-escaping) parameter to f.Params of the
// specified name, type and source position.
//
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
	param := f.addParam(name, obj.Type(), obj.Pos())
	param.object = obj
	return param
}

// addSpilledParam declares a parameter that is pre-spilled to the
// stack; the function body will load/store the spilled location.
// Subsequent lifting will eliminate spills where possible.
//
func (f *Function) addSpilledParam(obj types.Object) {
	param := f.addParamObj(obj)
	spill := &Alloc{Comment: obj.Name()}
	spill.setType(types.NewPointer(obj.Type()))
	spill.setPos(obj.Pos())
	f.objects[obj] = spill
	f.Locals = append(f.Locals, spill)
	f.emit(spill)
	f.emit(&Store{Addr: spill, Val: param})
}

// startBody initializes the function prior to generating SSA code for its body.
// Precondition: f.Type() already set.
//
func (f *Function) startBody() {
	f.currentBlock = f.newBasicBlock("entry")
	f.objects = make(map[types.Object]Value) // needed for some synthetics, e.g. init
}

// createSyntacticParams populates f.Params and generates code (spills
// and named result locals) for all the parameters declared in the
// syntax.  In addition it populates the f.objects mapping.
//
// Preconditions:
// f.startBody() was called.
// Postcondition:
// len(f.Params) == len(f.Signature.Params) + (f.Signature.Recv() ? 1 : 0)
//
func (f *Function) createSyntacticParams(recv *ast.FieldList, functype *ast.FuncType) {
	// Receiver (at most one inner iteration).
	if recv != nil {
		for _, field := range recv.List {
			for _, n := range field.Names {
				f.addSpilledParam(f.Pkg.objectOf(n))
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
				f.addSpilledParam(f.Pkg.objectOf(n))
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

// numberRegisters assigns numbers to all SSA registers
// (value-defining Instructions) in f, to aid debugging.
// (Non-Instruction Values are named at construction.)
//
func numberRegisters(f *Function) {
	v := 0
	for _, b := range f.Blocks {
		for _, instr := range b.Instrs {
			switch instr.(type) {
			case Value:
				instr.(interface {
					setNum(int)
				}).setNum(v)
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

// finishBody() finalizes the function after SSA code generation of its body.
func (f *Function) finishBody() {
	f.objects = nil
	f.currentBlock = nil
	f.lblocks = nil

	// Don't pin the AST in memory (except in debug mode).
	if n := f.syntax; n != nil && !f.debugInfo() {
		f.syntax = extentNode{n.Pos(), n.End()}
	}

	// Remove any f.Locals that are now heap-allocated.
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
		// f.DumpTo(os.Stderr)
		lift(f)
	}

	f.namedResults = nil // (used by lifting)

	numberRegisters(f)

	if f.Prog.mode&LogFunctions != 0 {
		f.DumpTo(os.Stderr)
	}

	if f.Prog.mode&SanityCheckFunctions != 0 {
		mustSanityCheck(f, nil)
	}
}

// removeNilBlocks eliminates nils from f.Blocks and updates each
// BasicBlock.Index.  Use this after any pass that may delete blocks.
//
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
//
func (pkg *Package) SetDebugMode(debug bool) {
	// TODO(adonovan): do we want ast.File granularity?
	pkg.debug = debug
}

// debugInfo reports whether debug info is wanted for this function.
func (f *Function) debugInfo() bool {
	return f.Pkg != nil && f.Pkg.debug
}

// addNamedLocal creates a local variable, adds it to function f and
// returns it.  Its name and type are taken from obj.  Subsequent
// calls to f.lookup(obj) will return the same local.
//
func (f *Function) addNamedLocal(obj types.Object) *Alloc {
	l := f.addLocal(obj.Type(), obj.Pos())
	l.Comment = obj.Name()
	f.objects[obj] = l
	return l
}

func (f *Function) addLocalForIdent(id *ast.Ident) *Alloc {
	return f.addNamedLocal(f.Pkg.objectOf(id))
}

// addLocal creates an anonymous local variable of type typ, adds it
// to function f and returns it.  pos is the optional source location.
//
func (f *Function) addLocal(typ types.Type, pos token.Pos) *Alloc {
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
//
func (f *Function) lookup(obj types.Object, escaping bool) Value {
	if v, ok := f.objects[obj]; ok {
		if alloc, ok := v.(*Alloc); ok && escaping {
			alloc.Heap = true
		}
		return v // function-local var (address)
	}

	// Definition must be in an enclosing function;
	// plumb it through intervening closures.
	if f.Enclosing == nil {
		panic("no Value for type.Object " + obj.Name())
	}
	outer := f.Enclosing.lookup(obj, true) // escaping
	v := &Capture{
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

// emit emits the specified instruction to function f, updating the
// control-flow graph if required.
//
func (f *Function) emit(instr Instruction) Value {
	return f.currentBlock.emit(instr)
}

// RelString returns the full name of this function, qualified by
// package name, receiver type, etc.
//
// The specific formatting rules are not guaranteed and may change.
//
// Examples:
//      "math.IsNaN"                // a package-level function
//      "IsNaN"                     // intra-package reference to same
//      "(*sync.WaitGroup).Add"     // a declared method
//      "(*Return).Block"           // a promotion wrapper method (intra-package ref)
//      "(Instruction).Block"       // an interface method wrapper (intra-package ref)
//      "func@5.32"                 // an anonymous function
//      "bound$(*T).f"              // a bound method wrapper
//
// If from==f.Pkg, suppress package qualification.
func (f *Function) RelString(from *types.Package) string {
	// TODO(adonovan): expose less fragile case discrimination
	// using f.method.

	// Anonymous?
	if f.Enclosing != nil {
		return f.name
	}

	// Declared method, or promotion/indirection wrapper?
	if recv := f.Signature.Recv(); recv != nil {
		return fmt.Sprintf("(%s).%s", relType(recv.Type(), from), f.name)
	}

	// Other synthetic wrapper?
	if f.Synthetic != "" {
		// Bound method wrapper?
		if strings.HasPrefix(f.name, "bound$") {
			return f.name
		}

		// Interface method wrapper?
		if strings.HasPrefix(f.Synthetic, "interface ") {
			return fmt.Sprintf("(%s).%s", relType(f.Params[0].Type(), from), f.name)
		}

		// "package initializer" or "loaded from GC object file": fall through.
	}

	// Package-level function.
	// Prefix with package name for cross-package references only.
	if p := f.pkgobj(); p != from {
		return fmt.Sprintf("%s.%s", p.Path(), f.name)
	}
	return f.name
}

// writeSignature writes to w the signature sig in declaration syntax.
// Derived from types.Signature.String().
//
func writeSignature(w io.Writer, pkg *types.Package, name string, sig *types.Signature, params []*Parameter) {
	io.WriteString(w, "func ")
	if recv := sig.Recv(); recv != nil {
		io.WriteString(w, "(")
		if n := params[0].Name(); n != "" {
			io.WriteString(w, n)
			io.WriteString(w, " ")
		}
		io.WriteString(w, relType(params[0].Type(), pkg))
		io.WriteString(w, ") ")
		params = params[1:]
	}
	io.WriteString(w, name)
	io.WriteString(w, "(")
	for i, v := range params {
		if i > 0 {
			io.WriteString(w, ", ")
		}
		io.WriteString(w, v.Name())
		io.WriteString(w, " ")
		if sig.IsVariadic() && i == len(params)-1 {
			io.WriteString(w, "...")
			io.WriteString(w, relType(v.Type().Underlying().(*types.Slice).Elem(), pkg))
		} else {
			io.WriteString(w, relType(v.Type(), pkg))
		}
	}
	io.WriteString(w, ")")
	if n := sig.Results().Len(); n > 0 {
		io.WriteString(w, " ")
		r := sig.Results()
		if n == 1 && r.At(0).Name() == "" {
			io.WriteString(w, relType(r.At(0).Type(), pkg))
		} else {
			io.WriteString(w, relType(r, pkg))
		}
	}
}

func (f *Function) pkgobj() *types.Package {
	if f.Pkg != nil {
		return f.Pkg.Object
	}
	return nil
}

// DumpTo prints to w a human readable "disassembly" of the SSA code of
// all basic blocks of function f.
//
func (f *Function) DumpTo(w io.Writer) {
	fmt.Fprintf(w, "# Name: %s\n", f.String())
	if f.Pkg != nil {
		fmt.Fprintf(w, "# Package: %s\n", f.Pkg.Object.Path())
	}
	if syn := f.Synthetic; syn != "" {
		fmt.Fprintln(w, "# Synthetic:", syn)
	}
	if pos := f.Pos(); pos.IsValid() {
		fmt.Fprintf(w, "# Location: %s\n", f.Prog.Fset.Position(pos))
	}

	if f.Enclosing != nil {
		fmt.Fprintf(w, "# Parent: %s\n", f.Enclosing.Name())
	}

	if f.Recover != nil {
		fmt.Fprintf(w, "# Recover: %s\n", f.Recover)
	}

	pkgobj := f.pkgobj()

	if f.FreeVars != nil {
		io.WriteString(w, "# Free variables:\n")
		for i, fv := range f.FreeVars {
			fmt.Fprintf(w, "# % 3d:\t%s %s\n", i, fv.Name(), relType(fv.Type(), pkgobj))
		}
	}

	if len(f.Locals) > 0 {
		io.WriteString(w, "# Locals:\n")
		for i, l := range f.Locals {
			fmt.Fprintf(w, "# % 3d:\t%s %s\n", i, l.Name(), relType(deref(l.Type()), pkgobj))
		}
	}
	writeSignature(w, pkgobj, f.Name(), f.Signature, f.Params)
	io.WriteString(w, ":\n")

	if f.Blocks == nil {
		io.WriteString(w, "\t(external)\n")
	}

	// NB. column calculations are confused by non-ASCII characters.
	const punchcard = 80 // for old time's sake.
	for _, b := range f.Blocks {
		if b == nil {
			// Corrupt CFG.
			fmt.Fprintf(w, ".nil:\n")
			continue
		}
		n, _ := fmt.Fprintf(w, ".%s:", b)
		fmt.Fprintf(w, "%*sP:%d S:%d\n", punchcard-1-n-len("P:n S:n"), "", len(b.Preds), len(b.Succs))

		if false { // CFG debugging
			fmt.Fprintf(w, "\t# CFG: %s --> %s --> %s\n", b.Preds, b, b.Succs)
		}
		for _, instr := range b.Instrs {
			io.WriteString(w, "\t")
			switch v := instr.(type) {
			case Value:
				l := punchcard
				// Left-align the instruction.
				if name := v.Name(); name != "" {
					n, _ := fmt.Fprintf(w, "%s = ", name)
					l -= n
				}
				// TODO(adonovan): append instructions directly to w.
				n, _ := io.WriteString(w, instr.String())
				l -= n
				// Right-align the type.
				if t := v.Type(); t != nil {
					fmt.Fprintf(w, " %*s", l-10, relType(t, pkgobj))
				}
			case nil:
				// Be robust against bad transforms.
				io.WriteString(w, "<deleted>")
			default:
				io.WriteString(w, instr.String())
			}
			io.WriteString(w, "\n")
		}
	}
	fmt.Fprintf(w, "\n")
}

// newBasicBlock adds to f a new basic block and returns it.  It does
// not automatically become the current block for subsequent calls to emit.
// comment is an optional string for more readable debugging output.
//
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

// NewFunction returns a new synthetic Function instance with its name
// and signature fields set as specified.
//
// The caller is responsible for initializing the remaining fields of
// the function object, e.g. Pkg, Prog, Params, Blocks.
//
// It is practically impossible for clients to construct well-formed
// SSA functions/packages/programs directly, so we assume this is the
// job of the Builder alone.  NewFunction exists to provide clients a
// little flexibility.  For example, analysis tools may wish to
// construct fake Functions for the root of the callgraph, a fake
// "reflect" package, etc.
//
// TODO(adonovan): think harder about the API here.
//
func NewFunction(name string, sig *types.Signature, provenance string) *Function {
	return &Function{name: name, Signature: sig, Synthetic: provenance}
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
//
func (f *Function) Syntax() ast.Node { return f.syntax }
