package ssa

// This file implements the Function and BasicBlock types.

import (
	"fmt"
	"go/ast"
	"go/types"
	"io"
	"os"
)

// Mode bits for additional diagnostics and checking.
// TODO(adonovan): move these to builder.go once submitted.
type BuilderMode uint

const (
	LogPackages          BuilderMode = 1 << iota // Dump package inventory to stderr
	LogFunctions                                 // Dump function SSA code to stderr
	LogSource                                    // Show source locations as SSA builder progresses
	SanityCheckFunctions                         // Perform sanity checking of function bodies
	UseGCImporter                                // Ignore SourceLoader; use gc-compiled object code for all imports
)

// addEdge adds a control-flow graph edge from from to to.
func addEdge(from, to *BasicBlock) {
	from.Succs = append(from.Succs, to)
	to.Preds = append(to.Preds, from)
}

// emit appends an instruction to the current basic block.
// If the instruction defines a Value, it is returned.
//
func (b *BasicBlock) emit(i Instruction) Value {
	i.SetBlock(b)
	b.Instrs = append(b.Instrs, i)
	v, _ := i.(Value)
	return v
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

// funcSyntax holds the syntax tree for the function declaration and body.
type funcSyntax struct {
	recvField    *ast.FieldList
	paramFields  *ast.FieldList
	resultFields *ast.FieldList
	body         *ast.BlockStmt
}

// labelledBlock returns the branch target associated with the
// specified label, creating it if needed.
//
func (f *Function) labelledBlock(label *ast.Ident) *lblock {
	lb := f.lblocks[label.Obj]
	if lb == nil {
		lb = &lblock{_goto: f.newBasicBlock("label." + label.Name)}
		f.lblocks[label.Obj] = lb
	}
	return lb
}

// addParam adds a (non-escaping) parameter to f.Params of the
// specified name and type.
//
func (f *Function) addParam(name string, typ types.Type) *Parameter {
	v := &Parameter{
		Name_: name,
		Type_: typ,
	}
	f.Params = append(f.Params, v)
	return v
}

// addSpilledParam declares a parameter that is pre-spilled to the
// stack; the function body will load/store the spilled location.
// Subsequent registerization will eliminate spills where possible.
//
func (f *Function) addSpilledParam(obj types.Object) {
	name := obj.GetName()
	param := f.addParam(name, obj.GetType())
	spill := &Alloc{
		Name_: name + "~", // "~" means "spilled"
		Type_: pointer(obj.GetType()),
	}
	f.objects[obj] = spill
	f.Locals = append(f.Locals, spill)
	f.emit(spill)
	f.emit(&Store{Addr: spill, Val: param})
}

// start initializes the function prior to generating SSA code for its body.
// Precondition: f.Type() already set.
//
// If f.syntax != nil, f is a Go source function and idents must be a
// mapping from syntactic identifiers to their canonical type objects;
// Otherwise, idents is ignored and the usual set-up for Go source
// functions is skipped.
//
func (f *Function) start(mode BuilderMode, idents map[*ast.Ident]types.Object) {
	if mode&LogSource != 0 {
		fmt.Fprintf(os.Stderr, "build function %s @ %s\n", f.FullName(), f.Prog.Files.Position(f.Pos))
	}
	f.currentBlock = f.newBasicBlock("entry")
	f.objects = make(map[types.Object]Value) // needed for some synthetics, e.g. init
	if f.syntax == nil {
		return // synthetic function; no syntax tree
	}
	f.lblocks = make(map[*ast.Object]*lblock)

	// Receiver (at most one inner iteration).
	if f.syntax.recvField != nil {
		for _, field := range f.syntax.recvField.List {
			for _, n := range field.Names {
				f.addSpilledParam(idents[n])
			}
			if field.Names == nil {
				f.addParam(f.Signature.Recv.Name, f.Signature.Recv.Type)
			}
		}
	}

	// Parameters.
	if f.syntax.paramFields != nil {
		for _, field := range f.syntax.paramFields.List {
			for _, n := range field.Names {
				f.addSpilledParam(idents[n])
			}
		}
	}

	// Results.
	if f.syntax.resultFields != nil {
		for _, field := range f.syntax.resultFields.List {
			// Implicit "var" decl of locals for named results.
			for _, n := range field.Names {
				f.results = append(f.results, f.addNamedLocal(idents[n]))
			}
		}
	}
}

// finish() finalizes the function after SSA code generation of its body.
func (f *Function) finish(mode BuilderMode) {
	f.objects = nil
	f.results = nil
	f.currentBlock = nil
	f.lblocks = nil
	f.syntax = nil

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

	// Ensure all value-defining Instructions have register names.
	// (Non-Instruction Values are named at construction.)
	tmp := 0
	for _, b := range f.Blocks {
		for _, instr := range b.Instrs {
			switch instr := instr.(type) {
			case *Alloc:
				// Local Allocs may already be named.
				if instr.Name_ == "" {
					instr.Name_ = fmt.Sprintf("t%d", tmp)
					tmp++
				}
			case Value:
				instr.(interface {
					setNum(int)
				}).setNum(tmp)
				tmp++
			}
		}
	}
	optimizeBlocks(f)

	if mode&LogFunctions != 0 {
		f.DumpTo(os.Stderr)
	}
	if mode&SanityCheckFunctions != 0 {
		MustSanityCheck(f, nil)
	}
	if mode&LogSource != 0 {
		fmt.Fprintf(os.Stderr, "build function %s done\n", f.FullName())
	}
}

// addNamedLocal creates a local variable, adds it to function f and
// returns it.  Its name and type are taken from obj.  Subsequent
// calls to f.lookup(obj) will return the same local.
//
// Precondition: f.syntax != nil (i.e. a Go source function).
//
func (f *Function) addNamedLocal(obj types.Object) *Alloc {
	l := f.addLocal(obj.GetType())
	l.Name_ = obj.GetName()
	f.objects[obj] = l
	return l
}

// addLocal creates an anonymous local variable of type typ, adds it
// to function f and returns it.
//
func (f *Function) addLocal(typ types.Type) *Alloc {
	v := &Alloc{Type_: pointer(typ)}
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
		if escaping {
			// Walk up the chain of Captures.
			x := v
			for {
				if c, ok := x.(*Capture); ok {
					x = c.Outer
				} else {
					break
				}
			}
			// By construction, all captures are ultimately Allocs in the
			// naive SSA form.  Parameters are pre-spilled to the stack.
			x.(*Alloc).Heap = true
		}
		return v // function-local var (address)
	}

	// Definition must be in an enclosing function;
	// plumb it through intervening closures.
	if f.Enclosing == nil {
		panic("no Value for type.Object " + obj.GetName())
	}
	v := &Capture{f.Enclosing.lookup(obj, true)} // escaping
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

// DumpTo prints to w a human readable "disassembly" of the SSA code of
// all basic blocks of function f.
//
func (f *Function) DumpTo(w io.Writer) {
	fmt.Fprintf(w, "# Name: %s\n", f.FullName())
	fmt.Fprintf(w, "# Declared at %s\n", f.Prog.Files.Position(f.Pos))
	fmt.Fprintf(w, "# Type: %s\n", f.Signature)

	if f.Enclosing != nil {
		fmt.Fprintf(w, "# Parent: %s\n", f.Enclosing.Name())
	}

	if f.FreeVars != nil {
		io.WriteString(w, "# Free variables:\n")
		for i, fv := range f.FreeVars {
			fmt.Fprintf(w, "# % 3d:\t%s %s\n", i, fv.Name(), fv.Type())
		}
	}

	params := f.Params
	if f.Signature.Recv != nil {
		fmt.Fprintf(w, "func (%s) %s(", params[0].Name(), f.Name())
		params = params[1:]
	} else {
		fmt.Fprintf(w, "func %s(", f.Name())
	}
	for i, v := range params {
		if i > 0 {
			io.WriteString(w, ", ")
		}
		io.WriteString(w, v.Name())
	}
	io.WriteString(w, "):\n")

	for _, b := range f.Blocks {
		if b == nil {
			// Corrupt CFG.
			fmt.Fprintf(w, ".nil:\n")
			continue
		}
		fmt.Fprintf(w, ".%s:\t\t\t\t\t\t\t       P:%d S:%d\n", b.Name, len(b.Preds), len(b.Succs))
		if false { // CFG debugging
			fmt.Fprintf(w, "\t# CFG: %s --> %s --> %s\n", blockNames(b.Preds), b.Name, blockNames(b.Succs))
		}
		for _, instr := range b.Instrs {
			io.WriteString(w, "\t")
			if v, ok := instr.(Value); ok {
				l := 80 // for old time's sake.
				// Left-align the instruction.
				if name := v.Name(); name != "" {
					n, _ := fmt.Fprintf(w, "%s = ", name)
					l -= n
				}
				n, _ := io.WriteString(w, instr.String())
				l -= n
				// Right-align the type.
				if t := v.Type(); t != nil {
					fmt.Fprintf(w, "%*s", l-9, t)
				}
			} else {
				io.WriteString(w, instr.String())
			}
			io.WriteString(w, "\n")
		}
	}
	fmt.Fprintf(w, "\n")
}

// newBasicBlock adds to f a new basic block with a unique name and
// returns it.  It does not automatically become the current block for
// subsequent calls to emit.
//
func (f *Function) newBasicBlock(name string) *BasicBlock {
	b := &BasicBlock{
		Name: fmt.Sprintf("%d.%s", len(f.Blocks), name),
		Func: f,
	}
	b.Succs = b.succs2[:0]
	f.Blocks = append(f.Blocks, b)
	return b
}
