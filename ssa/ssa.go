package ssa

// This package defines a high-level intermediate representation for
// Go programs using static single-assignment (SSA) form.

import (
	"fmt"
	"go/ast"
	"go/token"
	"sync"

	"code.google.com/p/go.tools/go/exact"
	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/importer"
)

// A Program is a partial or complete Go program converted to SSA form.
//
type Program struct {
	Files           *token.FileSet              // position information for the files of this Program [TODO: rename Fset]
	Packages        map[string]*Package         // all loaded Packages, keyed by import path [TODO rename packagesByPath]
	packages        map[*types.Package]*Package // all loaded Packages, keyed by object [TODO rename Packages]
	Builtins        map[types.Object]*Builtin   // all built-in functions, keyed by typechecker objects.
	concreteMethods map[*types.Func]*Function   // maps named concrete methods to their code
	mode            BuilderMode                 // set of mode bits for SSA construction

	methodsMu           sync.Mutex                // guards the following maps:
	methodSets          map[types.Type]MethodSet  // concrete method set each type [TODO(adonovan): de-dup]
	indirectionWrappers map[*Function]*Function   // func(*T) wrappers for T-methods
	boundMethodWrappers map[*Function]*Function   // wrappers for curried x.Method closures
	ifaceMethodWrappers map[*types.Func]*Function // wrappers for curried I.Method functions
}

// A Package is a single analyzed Go package containing Members for
// all package-level functions, variables, constants and types it
// declares.  These may be accessed directly via Members, or via the
// type-specific accessor methods Func, Type, Var and Const.
//
type Package struct {
	Prog    *Program               // the owning program
	Types   *types.Package         // the type checker's package object for this package [TODO rename Object]
	Members map[string]Member      // all package members keyed by name
	values  map[types.Object]Value // package-level vars and funcs, keyed by object
	Init    *Function              // the package's (concatenated) init function

	// The following fields are set transiently, then cleared
	// after building.
	started int32                 // atomically tested and set at start of build phase
	info    *importer.PackageInfo // package ASTs and type information
}

// A Member is a member of a Go package, implemented by *Constant,
// *Global, *Function, or *Type; they are created by package-level
// const, var, func and type declarations respectively.
//
type Member interface {
	Name() string       // the declared name of the package member
	String() string     // human-readable information about the value
	Pos() token.Pos     // position of member's declaration, if known
	Type() types.Type   // the type of the package member
	Token() token.Token // token.{VAR,FUNC,CONST,TYPE}
}

// An Id identifies the name of a field of a struct type, or the name
// of a method of an interface or a named type.
//
// For exported names, i.e. those beginning with a Unicode upper-case
// letter, a simple string is unambiguous.
//
// However, a method set or struct may contain multiple unexported
// names with identical spelling that are logically distinct because
// they originate in different packages.  Unexported names must
// therefore be disambiguated by their package too.
//
// The Pkg field of an Id is therefore nil iff the name is exported.
//
// This type is suitable for use as a map key because the equivalence
// relation == is consistent with identifier equality.
type Id struct {
	Pkg  *types.Package
	Name string
}

// A MethodSet contains all the methods for a particular type T.
// The method sets for T and *T are distinct entities.
//
// All methods in the method set for T have a receiver type of exactly
// T.  The method set of *T may contain synthetic indirection methods
// that wrap methods whose receiver type is T.
//
type MethodSet map[Id]*Function

// A Type is a Member of a Package representing a package-level named type.
//
// Type() returns a *types.Named.
//
type Type struct {
	Object *types.TypeName
}

// A Constant is a Member of Package representing a package-level
// constant value.
//
// Pos() returns the position of the declaring ast.ValueSpec.Names[*]
// identifier.
//
// NB: a Constant is not a Value; it contains a literal Value, which
// it augments with the name and position of its 'const' declaration.
//
// TODO(adonovan): if we decide to add a token.Pos to literal, we
// should then add a name too, and merge Constant and Literal.
// Experiment.
//
type Constant struct {
	name  string
	Value *Literal
	pos   token.Pos
}

// An SSA value that can be referenced by an instruction.
type Value interface {
	// Name returns the name of this value, and determines how
	// this Value appears when used as an operand of an
	// Instruction.
	//
	// This is the same as the source name for Parameters,
	// Builtins, Functions, Captures, Globals and some Allocs.
	// For literals, it is a representation of the literal's value
	// and type.  For all other Values this is the name of the
	// virtual register defined by the instruction.
	//
	// The name of an SSA Value is not semantically significant,
	// and may not even be unique within a function.
	Name() string

	// If this value is an Instruction, String returns its
	// disassembled form; otherwise it returns unspecified
	// human-readable information about the Value, such as its
	// kind, name and type.
	String() string

	// Type returns the type of this value.  Many instructions
	// (e.g. IndexAddr) change the behaviour depending on the
	// types of their operands.
	Type() types.Type

	// Referrers returns the list of instructions that have this
	// value as one of their operands; it may contain duplicates
	// if an instruction has a repeated operand.
	//
	// Referrers actually returns a pointer through which the
	// caller may perform mutations to the object's state.
	//
	// Referrers is currently only defined for the function-local
	// values Capture, Parameter and all value-defining instructions.
	// It returns nil for Function, Builtin, Literal and Global.
	//
	// Instruction.Operands contains the inverse of this relation.
	Referrers() *[]Instruction

	// Pos returns the location of the source construct that
	// gave rise to this value, or token.NoPos if it was not
	// explicit in the source.
	//
	// For each ast.Expr type, a particular field is designated as
	// the canonical location for the expression, e.g. the Lparen
	// for an *ast.CallExpr.  This enables us to find the value
	// corresponding to a given piece of source syntax.
	//
	Pos() token.Pos
}

// An Instruction is an SSA instruction that computes a new Value or
// has some effect.
//
// An Instruction that defines a value (e.g. BinOp) also implements
// the Value interface; an Instruction that only has an effect (e.g. Store)
// does not.
//
type Instruction interface {
	// String returns the disassembled form of this value.  e.g.
	//
	// Examples of Instructions that define a Value:
	// e.g.  "x + y"     (BinOp)
	//       "len([])"   (Call)
	// Note that the name of the Value is not printed.
	//
	// Examples of Instructions that do define (are) Values:
	// e.g.  "ret x"     (Ret)
	//       "*y = x"    (Store)
	//
	// (This separation is useful for some analyses which
	// distinguish the operation from the value it
	// defines. e.g. 'y = local int' is both an allocation of
	// memory 'local int' and a definition of a pointer y.)
	String() string

	// Parent returns the function to which this instruction
	// belongs.
	Parent() *Function

	// Block returns the basic block to which this instruction
	// belongs.
	Block() *BasicBlock

	// SetBlock sets the basic block to which this instruction
	// belongs.
	SetBlock(*BasicBlock)

	// Operands returns the operands of this instruction: the
	// set of Values it references.
	//
	// Specifically, it appends their addresses to rands, a
	// user-provided slice, and returns the resulting slice,
	// permitting avoidance of memory allocation.
	//
	// The operands are appended in undefined order; the addresses
	// are always non-nil but may point to a nil Value.  Clients
	// may store through the pointers, e.g. to effect a value
	// renaming.
	//
	// Value.Referrers is a subset of the inverse of this
	// relation.  (Referrers are not tracked for all types of
	// Values.)
	Operands(rands []*Value) []*Value

	// Pos returns the location of the source construct that
	// gave rise to this instruction, or token.NoPos if it was not
	// explicit in the source.
	//
	// For each ast.Expr type, a particular field is designated as
	// the canonical location for the expression, e.g. the Lparen
	// for an *ast.CallExpr.  This enables us to find the
	// instruction corresponding to a given piece of source
	// syntax.
	//
	Pos() token.Pos
}

// Function represents the parameters, results and code of a function
// or method.
//
// If Blocks is nil, this indicates an external function for which no
// Go source code is available.  In this case, Captures and Locals
// will be nil too.  Clients performing whole-program analysis must
// handle external functions specially.
//
// Functions are immutable values; they do not have addresses.
//
// Blocks[0] is the function entry point; block order is not otherwise
// semantically significant, though it may affect the readability of
// the disassembly.
//
// A nested function that refers to one or more lexically enclosing
// local variables ("free variables") has Capture parameters.  Such
// functions cannot be called directly but require a value created by
// MakeClosure which, via its Bindings, supplies values for these
// parameters.
//
// If the function is a method (Signature.Recv() != nil) then the first
// element of Params is the receiver parameter.
//
// Pos() returns the declaring ast.FuncLit.Type.Func or the position
// of the ast.FuncDecl.Name, if the function was explicit in the
// source.
//
// Type() returns the function's Signature.
//
type Function struct {
	name      string
	Signature *types.Signature
	pos       token.Pos

	Enclosing *Function    // enclosing function if anon; nil if global
	Pkg       *Package     // enclosing package for Go source functions; otherwise nil
	Prog      *Program     // enclosing program
	Params    []*Parameter // function parameters; for methods, includes receiver
	FreeVars  []*Capture   // free variables whose values must be supplied by closure
	Locals    []*Alloc
	Blocks    []*BasicBlock // basic blocks of the function; nil => external
	AnonFuncs []*Function   // anonymous functions directly beneath this one

	// The following fields are set transiently during building,
	// then cleared.
	currentBlock *BasicBlock             // where to emit code
	objects      map[types.Object]Value  // addresses of local variables
	namedResults []*Alloc                // tuple of named results
	syntax       *funcSyntax             // abstract syntax trees for Go source functions
	targets      *targets                // linked stack of branch targets
	lblocks      map[*ast.Object]*lblock // labelled blocks
}

// An SSA basic block.
//
// The final element of Instrs is always an explicit transfer of
// control (If, Jump, Ret or Panic).
//
// A block may contain no Instructions only if it is unreachable,
// i.e. Preds is nil.  Empty blocks are typically pruned.
//
// BasicBlocks and their Preds/Succs relation form a (possibly cyclic)
// graph independent of the SSA Value graph.  It is illegal for
// multiple edges to exist between the same pair of blocks.
//
// The order of Preds and Succs are significant (to Phi and If
// instructions, respectively).
//
type BasicBlock struct {
	Index        int            // index of this block within Func.Blocks
	Comment      string         // optional label; no semantic significance
	parent       *Function      // parent function
	Instrs       []Instruction  // instructions in order
	Preds, Succs []*BasicBlock  // predecessors and successors
	succs2       [2]*BasicBlock // initial space for Succs.
	dom          *domNode       // node in dominator tree; optional.
	gaps         int            // number of nil Instrs (transient).
	rundefers    int            // number of rundefers (transient)
}

// Pure values ----------------------------------------

// A Capture represents a free variable of the function to which it
// belongs.
//
// Captures are used to implement anonymous functions, whose free
// variables are lexically captured in a closure formed by
// MakeClosure.  The referent of such a capture is an Alloc or another
// Capture and is considered a potentially escaping heap address, with
// pointer type.
//
// Captures are also used to implement bound method closures.  Such a
// capture represents the receiver value and may be of any type that
// has concrete methods.
//
// Pos() returns the position of the value that was captured, which
// belongs to an enclosing function.
//
type Capture struct {
	name      string
	typ       types.Type
	pos       token.Pos
	parent    *Function
	referrers []Instruction

	// Transiently needed during building.
	outer Value // the Value captured from the enclosing context.
}

// A Parameter represents an input parameter of a function.
//
type Parameter struct {
	name      string
	typ       types.Type
	pos       token.Pos
	parent    *Function
	referrers []Instruction
}

// A Literal represents the value of a constant expression.
//
// It may have a nil, boolean, string or numeric (integer, fraction or
// complex) value, or a []byte or []rune conversion of a string
// literal.
//
// Literals may be of named types.  A literal's underlying type can be
// a basic type, possibly one of the "untyped" types, or a slice type
// whose elements' underlying type is byte or rune.  A nil literal can
// have any reference type: interface, map, channel, pointer, slice,
// or function---but not "untyped nil".
//
// All source-level constant expressions are represented by a Literal
// of equal type and value.
//
// Value holds the exact value of the literal, independent of its
// Type(), using the same representation as package go/exact uses for
// constants.
//
// Pos() returns token.NoPos.
//
// Example printed form:
// 	42:int
//	"hello":untyped string
//	3+4i:MyComplex
//
type Literal struct {
	typ   types.Type
	Value exact.Value
}

// A Global is a named Value holding the address of a package-level
// variable.
//
// Pos() returns the position of the ast.ValueSpec.Names[*]
// identifier.
//
type Global struct {
	name string
	typ  types.Type
	pos  token.Pos

	Pkg *Package

	// The following fields are set transiently during building,
	// then cleared.
	spec *ast.ValueSpec // explained at buildGlobal
}

// A Builtin represents a built-in function, e.g. len.
//
// Builtins are immutable values.  Builtins do not have addresses.
//
// Type() returns a *types.Builtin.
// Built-in functions may have polymorphic or variadic types that are
// not expressible in Go's type system.
//
type Builtin struct {
	Object *types.Func // canonical types.Universe object for this built-in
}

// Value-defining instructions  ----------------------------------------

// The Alloc instruction reserves space for a value of the given type,
// zero-initializes it, and yields its address.
//
// Alloc values are always addresses, and have pointer types, so the
// type of the allocated space is actually indirect(Type()).
//
// If Heap is false, Alloc allocates space in the function's
// activation record (frame); we refer to an Alloc(Heap=false) as a
// "local" alloc.  Each local Alloc returns the same address each time
// it is executed within the same activation; the space is
// re-initialized to zero.
//
// If Heap is true, Alloc allocates space in the heap, and returns; we
// refer to an Alloc(Heap=true) as a "new" alloc.  Each new Alloc
// returns a different address each time it is executed.
//
// When Alloc is applied to a channel, map or slice type, it returns
// the address of an uninitialized (nil) reference of that kind; store
// the result of MakeSlice, MakeMap or MakeChan in that location to
// instantiate these types.
//
// Pos() returns the ast.CompositeLit.Lbrace for a composite literal,
// or the ast.CallExpr.Lparen for a call to new() or for a call that
// allocates a varargs slice.
//
// Example printed form:
// 	t0 = local int
// 	t1 = new int
//
type Alloc struct {
	anInstruction
	name      string
	typ       types.Type
	Heap      bool
	pos       token.Pos
	referrers []Instruction
	index     int // dense numbering; for lifting
}

// The Phi instruction represents an SSA φ-node, which combines values
// that differ across incoming control-flow edges and yields a new
// value.  Within a block, all φ-nodes must appear before all non-φ
// nodes.
//
// Pos() returns the position of the && or || for short-circuit
// control-flow joins, or that of the *Alloc for φ-nodes inserted
// during SSA renaming.
//
// Example printed form:
// 	t2 = phi [0.start: t0, 1.if.then: t1, ...]
//
type Phi struct {
	Register
	Comment string  // a hint as to its purpose
	Edges   []Value // Edges[i] is value for Block().Preds[i]
}

// The Call instruction represents a function or method call.
//
// The Call instruction yields the function result, if there is
// exactly one, or a tuple (empty or len>1) whose components are
// accessed via Extract.
//
// See CallCommon for generic function call documentation.
//
// Pos() returns the ast.CallExpr.Lparen, if explicit in the source.
//
// Example printed form:
// 	t2 = println(t0, t1)
// 	t4 = t3()
// 	t7 = invoke t5.Println(...t6)
//
type Call struct {
	Register
	Call CallCommon
}

// The BinOp instruction yields the result of binary operation X Op Y.
//
// Pos() returns the ast.BinaryExpr.OpPos, if explicit in the source.
//
// Example printed form:
// 	t1 = t0 + 1:int
//
type BinOp struct {
	Register
	// One of:
	// ADD SUB MUL QUO REM          + - * / %
	// AND OR XOR SHL SHR AND_NOT   & | ^ << >> &~
	// EQL LSS GTR NEQ LEQ GEQ      == != < <= < >=
	Op   token.Token
	X, Y Value
}

// The UnOp instruction yields the result of Op X.
// ARROW is channel receive.
// MUL is pointer indirection (load).
// XOR is bitwise complement.
// SUB is negation.
//
// If CommaOk and Op=ARROW, the result is a 2-tuple of the value above
// and a boolean indicating the success of the receive.  The
// components of the tuple are accessed using Extract.
//
// Pos() returns the ast.UnaryExpr.OpPos, if explicit in the source,
//
// Example printed form:
// 	t0 = *x
// 	t2 = <-t1,ok
//
type UnOp struct {
	Register
	Op      token.Token // One of: NOT SUB ARROW MUL XOR ! - <- * ^
	X       Value
	CommaOk bool
}

// The ChangeType instruction applies to X a value-preserving type
// change to Type().
//
// Type changes are permitted:
//    - between a named type and its underlying type.
//    - between two named types of the same underlying type.
//    - between (possibly named) pointers to identical base types.
//    - between f(T) functions and (T) func f() methods.
//    - from a bidirectional channel to a read- or write-channel,
//      optionally adding/removing a name.
//
// This operation cannot fail dynamically.
//
// Pos() returns the ast.CallExpr.Lparen, if the instruction arose
// from an explicit conversion in the source.
//
// Example printed form:
// 	t1 = changetype *int <- IntPtr (t0)
//
type ChangeType struct {
	Register
	X Value
}

// The Convert instruction yields the conversion of value X to type
// Type().
//
// A conversion may change the value and representation of its operand.
// Conversions are permitted:
//    - between real numeric types.
//    - between complex numeric types.
//    - between string and []byte or []rune.
//    - from (Unicode) integer to (UTF-8) string.
// A conversion may imply a type name change also.
//
// This operation cannot fail dynamically.
//
// Conversions of untyped string/number/bool constants to a specific
// representation are eliminated during SSA construction.
//
// Pos() returns the ast.CallExpr.Lparen, if the instruction arose
// from an explicit conversion in the source.
//
// Example printed form:
// 	t1 = convert []byte <- string (t0)
//
type Convert struct {
	Register
	X Value
}

// ChangeInterface constructs a value of one interface type from a
// value of another interface type known to be assignable to it.
//
// This operation fails if the operand is nil.
// For all other operands, well-typedness ensures success.
// Use TypeAssert for interface conversions that are uncertain.
//
// Pos() returns the ast.CallExpr.Lparen if the instruction arose from
// an explicit T(e) conversion; the ast.TypeAssertExpr.Lparen if the
// instruction arose from an explicit e.(T) operation; or token.NoPos
// otherwise.
//
// Example printed form:
// 	t1 = change interface interface{} <- I (t0)
//
type ChangeInterface struct {
	Register
	X Value
}

// MakeInterface constructs an instance of an interface type from a
// value of a concrete type.
//
// Use Program.MethodSet(X.Type()) to find the method-set of X.
//
// To construct the zero value of an interface type T, use:
// 	&Literal{exact.MakeNil(), T}
//
// Pos() returns the ast.CallExpr.Lparen, if the instruction arose
// from an explicit conversion in the source.
//
// Example printed form:
// 	t1 = make interface{} <- int (42:int)
// 	t2 = make Stringer <- t0
//
type MakeInterface struct {
	Register
	X Value
}

// The MakeClosure instruction yields a closure value whose code is
// Fn and whose free variables' values are supplied by Bindings.
//
// Type() returns a (possibly named) *types.Signature.
//
// Pos() returns the ast.FuncLit.Type.Func for a function literal
// closure or the ast.SelectorExpr.Sel for a bound method closure.
//
// Example printed form:
// 	t0 = make closure anon@1.2 [x y z]
// 	t1 = make closure bound$(main.I).add [i]
//
type MakeClosure struct {
	Register
	Fn       Value   // always a *Function
	Bindings []Value // values for each free variable in Fn.FreeVars
}

// The MakeMap instruction creates a new hash-table-based map object
// and yields a value of kind map.
//
// Type() returns a (possibly named) *types.Map.
//
// Pos() returns the ast.CallExpr.Lparen, if created by make(map), or
// the ast.CompositeLit.Lbrack if created by a literal.
//
// Example printed form:
// 	t1 = make map[string]int t0
// 	t1 = make StringIntMap t0
//
type MakeMap struct {
	Register
	Reserve Value // initial space reservation; nil => default
}

// The MakeChan instruction creates a new channel object and yields a
// value of kind chan.
//
// Type() returns a (possibly named) *types.Chan.
//
// Pos() returns the ast.CallExpr.Lparen for the make(chan) that
// created it.
//
// Example printed form:
// 	t0 = make chan int 0
// 	t0 = make IntChan 0
//
type MakeChan struct {
	Register
	Size Value // int; size of buffer; zero => synchronous.
}

// The MakeSlice instruction yields a slice of length Len backed by a
// newly allocated array of length Cap.
//
// Both Len and Cap must be non-nil Values of integer type.
//
// (Alloc(types.Array) followed by Slice will not suffice because
// Alloc can only create arrays of statically known length.)
//
// Type() returns a (possibly named) *types.Slice.
//
// Pos() returns the ast.CallExpr.Lparen for the make([]T) that
// created it.
//
// Example printed form:
// 	t1 = make []string 1:int t0
// 	t1 = make StringSlice 1:int t0
//
type MakeSlice struct {
	Register
	Len Value
	Cap Value
}

// The Slice instruction yields a slice of an existing string, slice
// or *array X between optional integer bounds Low and High.
//
// Type() returns string if the type of X was string, otherwise a
// *types.Slice with the same element type as X.
//
// Pos() returns the ast.SliceExpr.Lbrack if created by a x[:] slice
// operation, the ast.CompositeLit.Lbrace if created by a literal, or
// NoPos if not explicit in the source (e.g. a variadic argument slice).
//
// Example printed form:
// 	t1 = slice t0[1:]
//
type Slice struct {
	Register
	X         Value // slice, string, or *array
	Low, High Value // either may be nil
}

// The FieldAddr instruction yields the address of Field of *struct X.
//
// The field is identified by its index within the field list of the
// struct type of X.
//
// Type() returns a (possibly named) *types.Pointer.
//
// Pos() returns the position of the ast.SelectorExpr.Sel for the
// field, if explicit in the source.
//
// Example printed form:
// 	t1 = &t0.name [#1]
//
type FieldAddr struct {
	Register
	X     Value // *struct
	Field int   // index into X.Type().(*types.Struct).Fields
}

// The Field instruction yields the Field of struct X.
//
// The field is identified by its index within the field list of the
// struct type of X; by using numeric indices we avoid ambiguity of
// package-local identifiers and permit compact representations.
//
// Pos() returns the position of the ast.SelectorExpr.Sel for the
// field, if explicit in the source.
//
// Example printed form:
// 	t1 = t0.name [#1]
//
type Field struct {
	Register
	X     Value // struct
	Field int   // index into X.Type().(*types.Struct).Fields
}

// The IndexAddr instruction yields the address of the element at
// index Index of collection X.  Index is an integer expression.
//
// The elements of maps and strings are not addressable; use Lookup or
// MapUpdate instead.
//
// Type() returns a (possibly named) *types.Pointer.
//
// Pos() returns the ast.IndexExpr.Lbrack for the index operation, if
// explicit in the source.
//
// Example printed form:
// 	t2 = &t0[t1]
//
type IndexAddr struct {
	Register
	X     Value // slice or *array,
	Index Value // numeric index
}

// The Index instruction yields element Index of array X.
//
// Pos() returns the ast.IndexExpr.Lbrack for the index operation, if
// explicit in the source.
//
// Example printed form:
// 	t2 = t0[t1]
//
type Index struct {
	Register
	X     Value // array
	Index Value // integer index
}

// The Lookup instruction yields element Index of collection X, a map
// or string.  Index is an integer expression if X is a string or the
// appropriate key type if X is a map.
//
// If CommaOk, the result is a 2-tuple of the value above and a
// boolean indicating the result of a map membership test for the key.
// The components of the tuple are accessed using Extract.
//
// Pos() returns the ast.IndexExpr.Lbrack, if explicit in the source.
//
// Example printed form:
// 	t2 = t0[t1]
// 	t5 = t3[t4],ok
//
type Lookup struct {
	Register
	X       Value // string or map
	Index   Value // numeric or key-typed index
	CommaOk bool  // return a value,ok pair
}

// SelectState is a helper for Select.
// It represents one goal state and its corresponding communication.
//
type SelectState struct {
	Dir  ast.ChanDir // direction of case
	Chan Value       // channel to use (for send or receive)
	Send Value       // value to send (for send)
}

// The Select instruction tests whether (or blocks until) one or more
// of the specified sent or received states is entered.
//
// It returns a triple (index int, recv interface{}, recvOk bool)
// whose components, described below, must be accessed via the Extract
// instruction.
//
// If Blocking, select waits until exactly one state holds, i.e. a
// channel becomes ready for the designated operation of sending or
// receiving; select chooses one among the ready states
// pseudorandomly, performs the send or receive operation, and sets
// 'index' to the index of the chosen channel.
//
// If !Blocking, select doesn't block if no states hold; instead it
// returns immediately with index equal to -1.
//
// If the chosen channel was used for a receive, 'recv' is set to the
// received value; otherwise it is nil.
//
// The third component of the triple, recvOk, is a boolean whose value
// is true iff the selected operation was a receive and the receive
// successfully yielded a value.
//
// Pos() returns the ast.SelectStmt.Select.
//
// Example printed form:
// 	t3 = select nonblocking [<-t0, t1<-t2, ...]
// 	t4 = select blocking []
//
type Select struct {
	Register
	States   []SelectState
	Blocking bool
}

// The Range instruction yields an iterator over the domain and range
// of X, which must be a string or map.
//
// Elements are accessed via Next.
//
// Type() returns a (possibly named) *types.Tuple.
//
// Pos() returns the ast.RangeStmt.For.
//
// Example printed form:
// 	t0 = range "hello":string
//
type Range struct {
	Register
	X Value // string or map
}

// The Next instruction reads and advances the (map or string)
// iterator Iter and returns a 3-tuple value (ok, k, v).  If the
// iterator is not exhausted, ok is true and k and v are the next
// elements of the domain and range, respectively.  Otherwise ok is
// false and k and v are undefined.
//
// Components of the tuple are accessed using Extract.
//
// The IsString field distinguishes iterators over strings from those
// over maps, as the Type() alone is insufficient: consider
// map[int]rune.
//
// Type() returns a *types.Tuple for the triple (ok, k, v).
// The types of k and/or v may be types.Invalid.
//
// Example printed form:
// 	t1 = next t0
//
type Next struct {
	Register
	Iter     Value
	IsString bool // true => string iterator; false => map iterator.
}

// The TypeAssert instruction tests whether interface value X has type
// AssertedType.
//
// If !CommaOk, on success it returns v, the result of the conversion
// (defined below); on failure it panics.
//
// If CommaOk: on success it returns a pair (v, true) where v is the
// result of the conversion; on failure it returns (z, false) where z
// is AssertedType's zero value.  The components of the pair must be
// accessed using the Extract instruction.
//
// If AssertedType is a concrete type, TypeAssert checks whether the
// dynamic type in interface X is equal to it, and if so, the result
// of the conversion is a copy of the value in the interface.
//
// If AssertedType is an interface, TypeAssert checks whether the
// dynamic type of the interface is assignable to it, and if so, the
// result of the conversion is a copy of the interface value X.
// If AssertedType is a superinterface of X.Type(), the operation
// cannot fail; ChangeInterface is preferred in this case.
//
// Type() reflects the actual type of the result, possibly a
// 2-types.Tuple; AssertedType is the asserted type.
//
// Pos() returns the ast.CallExpr.Lparen if the instruction arose from
// an explicit T(e) conversion; the ast.TypeAssertExpr.Lparen if the
// instruction arose from an explicit e.(T) operation; or token.NoPos
// otherwise.
//
// Example printed form:
// 	t1 = typeassert t0.(int)
// 	t3 = typeassert,ok t2.(T)
//
type TypeAssert struct {
	Register
	X            Value
	AssertedType types.Type
	CommaOk      bool
}

// The Extract instruction yields component Index of Tuple.
//
// This is used to access the results of instructions with multiple
// return values, such as Call, TypeAssert, Next, UnOp(ARROW) and
// IndexExpr(Map).
//
// Example printed form:
// 	t1 = extract t0 #1
//
type Extract struct {
	Register
	Tuple Value
	Index int
}

// Instructions executed for effect.  They do not yield a value. --------------------

// The Jump instruction transfers control to the sole successor of its
// owning block.
//
// A Jump must be the last instruction of its containing BasicBlock.
//
// Pos() returns NoPos.
//
// Example printed form:
// 	jump done
//
type Jump struct {
	anInstruction
}

// The If instruction transfers control to one of the two successors
// of its owning block, depending on the boolean Cond: the first if
// true, the second if false.
//
// An If instruction must be the last instruction of its containing
// BasicBlock.
//
// Pos() returns NoPos.
//
// Example printed form:
// 	if t0 goto done else body
//
type If struct {
	anInstruction
	Cond Value
}

// The Ret instruction returns values and control back to the calling
// function.
//
// len(Results) is always equal to the number of results in the
// function's signature.
//
// If len(Results) > 1, Ret returns a tuple value with the specified
// components which the caller must access using Extract instructions.
//
// There is no instruction to return a ready-made tuple like those
// returned by a "value,ok"-mode TypeAssert, Lookup or UnOp(ARROW) or
// a tail-call to a function with multiple result parameters.
//
// Ret must be the last instruction of its containing BasicBlock.
// Such a block has no successors.
//
// Pos() returns the ast.ReturnStmt.Return, if explicit in the source.
//
// Example printed form:
// 	ret
// 	ret nil:I, 2:int
//
type Ret struct {
	anInstruction
	Results []Value
	pos     token.Pos
}

// The RunDefers instruction pops and invokes the entire stack of
// procedure calls pushed by Defer instructions in this function.
//
// It is legal to encounter multiple 'rundefers' instructions in a
// single control-flow path through a function; this is useful in
// the combined init() function, for example.
//
// Pos() returns NoPos.
//
// Example printed form:
//	rundefers
//
type RunDefers struct {
	anInstruction
}

// The Panic instruction initiates a panic with value X.
//
// A Panic instruction must be the last instruction of its containing
// BasicBlock, which must have no successors.
//
// NB: 'go panic(x)' and 'defer panic(x)' do not use this instruction;
// they are treated as calls to a built-in function.
//
// Pos() returns the ast.CallExpr.Lparen if this panic was explicit
// in the source.
//
// Example printed form:
// 	panic t0
//
type Panic struct {
	anInstruction
	X   Value // an interface{}
	pos token.Pos
}

// The Go instruction creates a new goroutine and calls the specified
// function within it.
//
// See CallCommon for generic function call documentation.
//
// Example printed form:
// 	go println(t0, t1)
// 	go t3()
// 	go invoke t5.Println(...t6)
//
type Go struct {
	anInstruction
	Call CallCommon
}

// The Defer instruction pushes the specified call onto a stack of
// functions to be called by a RunDefers instruction or by a panic.
//
// See CallCommon for generic function call documentation.
//
// Example printed form:
// 	defer println(t0, t1)
// 	defer t3()
// 	defer invoke t5.Println(...t6)
//
type Defer struct {
	anInstruction
	Call CallCommon
}

// The Send instruction sends X on channel Chan.
//
// Pos() returns the ast.SendStmt.Arrow, if explicit in the source.
//
// Example printed form:
// 	send t0 <- t1
//
type Send struct {
	anInstruction
	Chan, X Value
	pos     token.Pos
}

// The Store instruction stores Val at address Addr.
// Stores can be of arbitrary types.
//
// Pos() returns the ast.StarExpr.Star, if explicit in the source.
//
// Example printed form:
// 	*x = y
//
type Store struct {
	anInstruction
	Addr Value
	Val  Value
	pos  token.Pos
}

// The MapUpdate instruction updates the association of Map[Key] to
// Value.
//
// Pos() returns the ast.KeyValueExpr.Colon, if explicit in the source.
//
// Example printed form:
//	t0[t1] = t2
//
type MapUpdate struct {
	anInstruction
	Map   Value
	Key   Value
	Value Value
	pos   token.Pos
}

// Embeddable mix-ins and helpers for common parts of other structs. -----------

// Register is a mix-in embedded by all SSA values that are also
// instructions, i.e. virtual registers, and provides implementations
// of the Value interface's Name() and Type() methods: the name is
// simply a numbered register (e.g. "t0") and the type is the Type_
// field.
//
// Temporary names are automatically assigned to each Register on
// completion of building a function in SSA form.
//
// Clients must not assume that the 'id' value (and the Name() derived
// from it) is unique within a function.  As always in this API,
// semantics are determined only by identity; names exist only to
// facilitate debugging.
//
type Register struct {
	anInstruction
	num       int        // "name" of virtual register, e.g. "t0".  Not guaranteed unique.
	typ       types.Type // type of virtual register
	pos       token.Pos  // position of source expression, or NoPos
	referrers []Instruction
}

// anInstruction is a mix-in embedded by all Instructions.
// It provides the implementations of the Block and SetBlock methods.
type anInstruction struct {
	block *BasicBlock // the basic block of this instruction
}

// CallCommon is contained by Go, Defer and Call to hold the
// common parts of a function or method call.
//
// Each CallCommon exists in one of two modes, function call and
// interface method invocation, or "call" and "invoke" for short.
//
// 1. "call" mode: when Recv is nil (!IsInvoke), a CallCommon
// represents an ordinary function call of the value in Func.
//
// In the common case in which Func is a *Function, this indicates a
// statically dispatched call to a package-level function, an
// anonymous function, or a method of a named type.  Also statically
// dispatched, but less common, Func may be a *MakeClosure, indicating
// an immediately applied function literal with free variables.  Any
// other Value of Func indicates a dynamically dispatched function
// call.  The StaticCallee method returns the callee in these cases.
//
// Args contains the arguments to the call.  If Func is a method,
// Args[0] contains the receiver parameter.  Recv and Method are not
// used in this mode.
//
// Example printed form:
// 	t2 = println(t0, t1)
// 	go t3()
//	defer t5(...t6)
//
// 2. "invoke" mode: when Recv is non-nil (IsInvoke), a CallCommon
// represents a dynamically dispatched call to an interface method.
// In this mode, Recv is the interface value and Method is the index
// of the method within the interface type of the receiver.
//
// Recv is implicitly supplied to the concrete method implementation
// as the receiver parameter; in other words, Args[0] holds not the
// receiver but the first true argument.  Func is not used in this
// mode.
//
// Example printed form:
// 	t1 = invoke t0.String()
// 	go invoke t3.Run(t2)
// 	defer invoke t4.Handle(...t5)
//
// In both modes, HasEllipsis is true iff the last element of Args is
// a slice value containing zero or more arguments to a variadic
// function.  (This is not semantically significant since the type of
// the called function is sufficient to determine this, but it aids
// readability of the printed form.)
//
type CallCommon struct {
	Recv        Value     // receiver, iff interface method invocation
	Method      int       // index of interface method; call MethodId() for its Id
	Func        Value     // target of call, iff function call
	Args        []Value   // actual parameters, including receiver in invoke mode
	HasEllipsis bool      // true iff last Args is a slice of '...' args (needed?)
	pos         token.Pos // position of CallExpr.Lparen, iff explicit in source
}

// IsInvoke returns true if this call has "invoke" (not "call") mode.
func (c *CallCommon) IsInvoke() bool {
	return c.Recv != nil
}

func (c *CallCommon) Pos() token.Pos { return c.pos }

// StaticCallee returns the called function if this is a trivially
// static "call"-mode call.
func (c *CallCommon) StaticCallee() *Function {
	switch fn := c.Func.(type) {
	case *Function:
		return fn
	case *MakeClosure:
		return fn.Fn.(*Function)
	}
	return nil
}

// MethodId returns the Id for the method called by c, which must
// have "invoke" mode.
func (c *CallCommon) MethodId() Id {
	m := c.Recv.Type().Underlying().(*types.Interface).Method(c.Method)
	return MakeId(m.Name(), m.Pkg())
}

// Description returns a description of the mode of this call suitable
// for a user interface, e.g. "static method call".
func (c *CallCommon) Description() string {
	switch fn := c.Func.(type) {
	case nil:
		return "dynamic method call" // ("invoke" mode)
	case *MakeClosure:
		return "static function closure call"
	case *Function:
		if fn.Signature.Recv() != nil {
			return "static method call"
		}
		return "static function call"
	}
	return "dynamic function call"
}

func (v *Builtin) Type() types.Type        { return v.Object.Type() }
func (v *Builtin) Name() string            { return v.Object.Name() }
func (*Builtin) Referrers() *[]Instruction { return nil }
func (v *Builtin) Pos() token.Pos          { return token.NoPos }

func (v *Capture) Type() types.Type          { return v.typ }
func (v *Capture) Name() string              { return v.name }
func (v *Capture) Referrers() *[]Instruction { return &v.referrers }
func (v *Capture) Pos() token.Pos            { return v.pos }
func (v *Capture) Parent() *Function         { return v.parent }

func (v *Global) Type() types.Type        { return v.typ }
func (v *Global) Name() string            { return v.name }
func (v *Global) Pos() token.Pos          { return v.pos }
func (*Global) Referrers() *[]Instruction { return nil }
func (v *Global) Token() token.Token      { return token.VAR }

func (v *Function) Name() string            { return v.name }
func (v *Function) Type() types.Type        { return v.Signature }
func (v *Function) Pos() token.Pos          { return v.pos }
func (*Function) Referrers() *[]Instruction { return nil }
func (v *Function) Token() token.Token      { return token.FUNC }

func (v *Parameter) Type() types.Type          { return v.typ }
func (v *Parameter) Name() string              { return v.name }
func (v *Parameter) Referrers() *[]Instruction { return &v.referrers }
func (v *Parameter) Pos() token.Pos            { return v.pos }
func (v *Parameter) Parent() *Function         { return v.parent }

func (v *Alloc) Type() types.Type          { return v.typ }
func (v *Alloc) Name() string              { return v.name }
func (v *Alloc) Referrers() *[]Instruction { return &v.referrers }
func (v *Alloc) Pos() token.Pos            { return v.pos }

func (v *Register) Type() types.Type          { return v.typ }
func (v *Register) setType(typ types.Type)    { v.typ = typ }
func (v *Register) Name() string              { return fmt.Sprintf("t%d", v.num) }
func (v *Register) setNum(num int)            { v.num = num }
func (v *Register) Referrers() *[]Instruction { return &v.referrers }
func (v *Register) asRegister() *Register     { return v }
func (v *Register) Pos() token.Pos            { return v.pos }
func (v *Register) setPos(pos token.Pos)      { v.pos = pos }

func (v *anInstruction) Parent() *Function          { return v.block.parent }
func (v *anInstruction) Block() *BasicBlock         { return v.block }
func (v *anInstruction) SetBlock(block *BasicBlock) { v.block = block }

func (t *Type) Name() string       { return t.Object.Name() }
func (t *Type) Pos() token.Pos     { return t.Object.Pos() }
func (t *Type) String() string     { return t.Name() }
func (t *Type) Type() types.Type   { return t.Object.Type() }
func (t *Type) Token() token.Token { return token.TYPE }

func (c *Constant) Name() string       { return c.name }
func (c *Constant) Pos() token.Pos     { return c.pos }
func (c *Constant) String() string     { return c.Name() }
func (c *Constant) Type() types.Type   { return c.Value.Type() }
func (c *Constant) Token() token.Token { return token.CONST }

// Func returns the package-level function of the specified name,
// or nil if not found.
//
func (p *Package) Func(name string) (f *Function) {
	f, _ = p.Members[name].(*Function)
	return
}

// Var returns the package-level variable of the specified name,
// or nil if not found.
//
func (p *Package) Var(name string) (g *Global) {
	g, _ = p.Members[name].(*Global)
	return
}

// Const returns the package-level constant of the specified name,
// or nil if not found.
//
func (p *Package) Const(name string) (c *Constant) {
	c, _ = p.Members[name].(*Constant)
	return
}

// Type returns the package-level type of the specified name,
// or nil if not found.
//
func (p *Package) Type(name string) (t *Type) {
	t, _ = p.Members[name].(*Type)
	return
}

// Value returns the program-level value corresponding to the
// specified named object, which may be a universal built-in
// (*Builtin) or a package-level var (*Global) or func (*Function) of
// some package in prog.  It returns nil if the object is not found.
//
func (prog *Program) Value(obj types.Object) Value {
	if p := obj.Pkg(); p != nil {
		if pkg, ok := prog.packages[p]; ok {
			return pkg.values[obj]
		}
		return nil
	}
	return prog.Builtins[obj]
}

// Package returns the SSA package corresponding to the specified
// type-checker package object.
// It returns nil if no such SSA package has been created.
//
func (prog *Program) Package(pkg *types.Package) *Package {
	return prog.packages[pkg]
}

func (v *Call) Pos() token.Pos      { return v.Call.pos }
func (s *Defer) Pos() token.Pos     { return s.Call.pos }
func (s *Go) Pos() token.Pos        { return s.Call.pos }
func (s *MapUpdate) Pos() token.Pos { return s.pos }
func (s *Panic) Pos() token.Pos     { return s.pos }
func (s *Ret) Pos() token.Pos       { return s.pos }
func (s *Send) Pos() token.Pos      { return s.pos }
func (s *Store) Pos() token.Pos     { return s.pos }
func (s *If) Pos() token.Pos        { return token.NoPos }
func (s *Jump) Pos() token.Pos      { return token.NoPos }
func (s *RunDefers) Pos() token.Pos { return token.NoPos }

// Operands.

func (v *Alloc) Operands(rands []*Value) []*Value {
	return rands
}

func (v *BinOp) Operands(rands []*Value) []*Value {
	return append(rands, &v.X, &v.Y)
}

func (c *CallCommon) Operands(rands []*Value) []*Value {
	rands = append(rands, &c.Recv, &c.Func)
	for i := range c.Args {
		rands = append(rands, &c.Args[i])
	}
	return rands
}

func (s *Go) Operands(rands []*Value) []*Value {
	return s.Call.Operands(rands)
}

func (s *Call) Operands(rands []*Value) []*Value {
	return s.Call.Operands(rands)
}

func (s *Defer) Operands(rands []*Value) []*Value {
	return s.Call.Operands(rands)
}

func (v *ChangeInterface) Operands(rands []*Value) []*Value {
	return append(rands, &v.X)
}

func (v *ChangeType) Operands(rands []*Value) []*Value {
	return append(rands, &v.X)
}

func (v *Convert) Operands(rands []*Value) []*Value {
	return append(rands, &v.X)
}

func (v *Extract) Operands(rands []*Value) []*Value {
	return append(rands, &v.Tuple)
}

func (v *Field) Operands(rands []*Value) []*Value {
	return append(rands, &v.X)
}

func (v *FieldAddr) Operands(rands []*Value) []*Value {
	return append(rands, &v.X)
}

func (s *If) Operands(rands []*Value) []*Value {
	return append(rands, &s.Cond)
}

func (v *Index) Operands(rands []*Value) []*Value {
	return append(rands, &v.X, &v.Index)
}

func (v *IndexAddr) Operands(rands []*Value) []*Value {
	return append(rands, &v.X, &v.Index)
}

func (*Jump) Operands(rands []*Value) []*Value {
	return rands
}

func (v *Lookup) Operands(rands []*Value) []*Value {
	return append(rands, &v.X, &v.Index)
}

func (v *MakeChan) Operands(rands []*Value) []*Value {
	return append(rands, &v.Size)
}

func (v *MakeClosure) Operands(rands []*Value) []*Value {
	rands = append(rands, &v.Fn)
	for i := range v.Bindings {
		rands = append(rands, &v.Bindings[i])
	}
	return rands
}

func (v *MakeInterface) Operands(rands []*Value) []*Value {
	return append(rands, &v.X)
}

func (v *MakeMap) Operands(rands []*Value) []*Value {
	return append(rands, &v.Reserve)
}

func (v *MakeSlice) Operands(rands []*Value) []*Value {
	return append(rands, &v.Len, &v.Cap)
}

func (v *MapUpdate) Operands(rands []*Value) []*Value {
	return append(rands, &v.Map, &v.Key, &v.Value)
}

func (v *Next) Operands(rands []*Value) []*Value {
	return append(rands, &v.Iter)
}

func (s *Panic) Operands(rands []*Value) []*Value {
	return append(rands, &s.X)
}

func (v *Phi) Operands(rands []*Value) []*Value {
	for i := range v.Edges {
		rands = append(rands, &v.Edges[i])
	}
	return rands
}

func (v *Range) Operands(rands []*Value) []*Value {
	return append(rands, &v.X)
}

func (s *Ret) Operands(rands []*Value) []*Value {
	for i := range s.Results {
		rands = append(rands, &s.Results[i])
	}
	return rands
}

func (*RunDefers) Operands(rands []*Value) []*Value {
	return rands
}

func (v *Select) Operands(rands []*Value) []*Value {
	for i := range v.States {
		rands = append(rands, &v.States[i].Chan, &v.States[i].Send)
	}
	return rands
}

func (s *Send) Operands(rands []*Value) []*Value {
	return append(rands, &s.Chan, &s.X)
}

func (v *Slice) Operands(rands []*Value) []*Value {
	return append(rands, &v.X, &v.Low, &v.High)
}

func (s *Store) Operands(rands []*Value) []*Value {
	return append(rands, &s.Addr, &s.Val)
}

func (v *TypeAssert) Operands(rands []*Value) []*Value {
	return append(rands, &v.X)
}

func (v *UnOp) Operands(rands []*Value) []*Value {
	return append(rands, &v.X)
}
