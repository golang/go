package ssa

// This package defines a high-level intermediate representation for
// Go programs using static single-assignment (SSA) form.

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
)

// A Program is a partial or complete Go program converted to SSA form.
// Each Builder creates and populates a single Program during its
// lifetime.
//
// TODO(adonovan): synthetic methods for promoted methods and for
// standalone interface methods do not belong to any package.  Make
// them enumerable here.
//
// TODO(adonovan): MethodSets of types other than named types
// (i.e. anon structs) are not currently accessible, nor are they
// memoized.  Add a method: MethodSetForType() which looks in the
// appropriate Package (for methods of named types) or in
// Program.AnonStructMethods (for methods of anon structs).
//
type Program struct {
	Files    *token.FileSet            // position information for the files of this Program
	Packages map[string]*Package       // all loaded Packages, keyed by import path
	Builtins map[types.Object]*Builtin // all built-in functions, keyed by typechecker objects.
}

// A Package is a single analyzed Go package, containing Members for
// all package-level functions, variables, constants and types it
// declares.  These may be accessed directly via Members, or via the
// type-specific accessor methods Func, Type, Var and Const.
//
type Package struct {
	Prog       *Program          // the owning program
	Types      *types.Package    // the type checker's package object for this package.
	ImportPath string            // e.g. "sync/atomic"
	Pos        token.Pos         // position of an arbitrary file in the package
	Members    map[string]Member // all exported and unexported members of the package
	AnonFuncs  []*Function       // all anonymous functions in this package
	Init       *Function         // the package's (concatenated) init function

	// The following fields are set transiently during building,
	// then cleared.
	files []*ast.File // the abstract syntax tree for the files of the package
}

// A Member is a member of a Go package, implemented by *Literal,
// *Global, *Function, or *Type; they are created by package-level
// const, var, func and type declarations respectively.
//
type Member interface {
	Name() string      // the declared name of the package member
	String() string    // human-readable information about the value
	Type() types.Type  // the type of the package member
	ImplementsMember() // dummy method to indicate the "implements" relation.
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

// A MethodSet contains all the methods whose receiver is either T or
// *T, for some named or struct type T.
//
// TODO(adonovan): the client is required to adapt T<=>*T, e.g. when
// invoking an interface method.  (This could be simplified for the
// client by having distinct method sets for T and *T, with the SSA
// Builder generating wrappers as needed, but probably the client is
// able to do a better job.)  Document the precise rules the client
// must follow.
//
type MethodSet map[Id]*Function

// A Type is a Member of a Package representing the name, underlying
// type and method set of a named type declared at package scope.
//
// The method set contains only concrete methods; it is empty for
// interface types.
//
type Type struct {
	NamedType *types.NamedType
	Methods   MethodSet
}

// An SSA value that can be referenced by an instruction.
//
// TODO(adonovan): add methods:
// - Referrers() []*Instruction // all instructions that refer to this value.
//
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
	//
	// Documented type invariants below (e.g. "Alloc.Type()
	// returns a *types.Pointer") refer to the underlying type in
	// the case of NamedTypes.
	Type() types.Type

	// Dummy method to indicate the "implements" relation.
	ImplementsValue()
}

// An Instruction is an SSA instruction that computes a new Value or
// has some effect.
//
// An Instruction that defines a value (e.g. BinOp) also implements
// the Value interface; an Instruction that only has an effect (e.g. Store)
// does not.
//
// TODO(adonovan): add method:
// - Operands() []Value  // all Values referenced by this instruction.
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

	// Block returns the basic block to which this instruction
	// belongs.
	Block() *BasicBlock

	// SetBlock sets the basic block to which this instruction
	// belongs.
	SetBlock(*BasicBlock)

	// Dummy method to indicate the "implements" relation.
	ImplementsInstruction()
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
// parameters.  Captures are always addresses.
//
// If the function is a method (Signature.Recv != nil) then the first
// element of Params is the receiver parameter.
//
// Type() returns the function's Signature.
//
type Function struct {
	Name_     string
	Signature *types.Signature

	Pos       token.Pos // location of the definition
	Enclosing *Function // enclosing function if anon; nil if global
	Pkg       *Package  // enclosing package; nil for some synthetic methods
	Prog      *Program  // enclosing program
	Params    []*Parameter
	FreeVars  []*Capture // free variables whose values must be supplied by closure
	Locals    []*Alloc
	Blocks    []*BasicBlock // basic blocks of the function; nil => external

	// The following fields are set transiently during building,
	// then cleared.
	currentBlock *BasicBlock             // where to emit code
	objects      map[types.Object]Value  // addresses of local variables
	results      []*Alloc                // tuple of named results
	syntax       *funcSyntax             // abstract syntax trees for Go source functions
	targets      *targets                // linked stack of branch targets
	lblocks      map[*ast.Object]*lblock // labelled blocks
}

// An SSA basic block.
//
// The final element of Instrs is always an explicit transfer of
// control (If, Jump or Ret).
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
	Name         string         // label; no semantic significance
	Func         *Function      // containing function
	Instrs       []Instruction  // instructions in order
	Preds, Succs []*BasicBlock  // predecessors and successors
	succs2       [2]*BasicBlock // initial space for Succs.
}

// Pure values ----------------------------------------

// A Capture is a pointer to a lexically enclosing local variable.
//
// The referent of a capture is an Alloc or another Capture and is
// always considered potentially escaping, so Captures are always
// addresses in the heap, and have pointer types.
//
type Capture struct {
	Outer Value // the Value captured from the enclosing context.
}

// A Parameter represents an input parameter of a function.
//
type Parameter struct {
	Name_ string
	Type_ types.Type
}

// A Literal represents a literal nil, boolean, string or numeric
// (integer, fraction or complex) value.
//
// A literal's underlying Type() can be a basic type, possibly one of
// the "untyped" types.  A nil literal can have any reference type:
// interface, map, channel, pointer, slice, or function---but not
// "untyped nil".
//
// All source-level constant expressions are represented by a Literal
// of equal type and value.
//
// Value holds the exact value of the literal, independent of its
// Type(), using the same representation as package go/types uses for
// constants.
//
// Example printed form:
// 	42:int
//	"hello":untyped string
//	3+4i:MyComplex
//
type Literal struct {
	Type_ types.Type
	Value interface{}
}

// A Global is a named Value holding the address of a package-level
// variable.
//
type Global struct {
	Name_ string
	Type_ types.Type
	Pkg   *Package

	// The following fields are set transiently during building,
	// then cleared.
	spec *ast.ValueSpec // explained at buildGlobal
}

// A built-in function, e.g. len.
//
// Builtins are immutable values; they do not have addresses.
//
// Type() returns an inscrutable *types.builtin.  Built-in functions
// may have polymorphic or variadic types that are not expressible in
// Go's type system.
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
// Example printed form:
// 	t0 = local int
// 	t1 = new int
//
type Alloc struct {
	anInstruction
	Name_ string
	Type_ types.Type
	Heap  bool
}

// Phi represents an SSA φ-node, which combines values that differ
// across incoming control-flow edges and yields a new value.  Within
// a block, all φ-nodes must appear before all non-φ nodes.
//
// Example printed form:
// 	t2 = phi [0.start: t0, 1.if.then: t1, ...]
//
type Phi struct {
	Register
	Edges []Value // Edges[i] is value for Block().Preds[i]
}

// Call represents a function or method call.
//
// The Call instruction yields the function result, if there is
// exactly one, or a tuple (empty or len>1) whose components are
// accessed via Extract.
//
// See CallCommon for generic function call documentation.
//
// Example printed form:
// 	t2 = println(t0, t1)
// 	t4 = t3()
// 	t7 = invoke t5.Println(...t6)
//
type Call struct {
	Register
	CallCommon
}

// BinOp yields the result of binary operation X Op Y.
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

// UnOp yields the result of Op X.
// ARROW is channel receive.
// MUL is pointer indirection (load).
//
// If CommaOk and Op=ARROW, the result is a 2-tuple of the value above
// and a boolean indicating the success of the receive.  The
// components of the tuple are accessed using Extract.
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

// Conv yields the conversion of X to type Type().
//
// A conversion is one of the following kinds.  The behaviour of the
// conversion operator may depend on both Type() and X.Type(), as well
// as the dynamic value.
//
// A '+' indicates that a dynamic representation change may occur.
// A '-' indicates that the conversion is a value-preserving change
// to types only.
//
// 1. implicit conversions (arising from assignability rules):
//    - adding/removing a name, same underlying types.
//    - channel type restriction, possibly adding/removing a name.
// 2. explicit conversions (in addition to the above):
//    - changing a name, same underlying types.
//    - between pointers to identical base types.
//    + conversions between real numeric types.
//    + conversions between complex numeric types.
//    + integer/[]byte/[]rune -> string.
//    + string -> []byte/[]rune.
//
// TODO(adonovan): split into two cases:
// - rename value (ChangeType)
// + value to type with different representation (Conv)
//
// Conversions of untyped string/number/bool constants to a specific
// representation are eliminated during SSA construction.
//
// Example printed form:
// 	t1 = convert interface{} <- int (t0)
//
type Conv struct {
	Register
	X Value
}

// ChangeInterface constructs a value of one interface type from a
// value of another interface type known to be assignable to it.
//
// Example printed form:
// 	t1 = change interface interface{} <- I (t0)
//
type ChangeInterface struct {
	Register
	X Value
}

// MakeInterface constructs an instance of an interface type from a
// value and its method-set.
//
// To construct the zero value of an interface type T, use:
// 	&Literal{types.nilType{}, T}
//
// Example printed form:
// 	t1 = make interface interface{} <- int (42:int)
//
type MakeInterface struct {
	Register
	X       Value
	Methods MethodSet // method set of (non-interface) X iff converting to interface
}

// A MakeClosure instruction yields an anonymous function value whose
// code is Fn and whose lexical capture slots are populated by Bindings.
//
// By construction, all captured variables are addresses of variables
// allocated with 'new', i.e. Alloc(Heap=true).
//
// Type() returns a *types.Signature.
//
// Example printed form:
// 	t0 = make closure anon@1.2 [x y z]
//
type MakeClosure struct {
	Register
	Fn       *Function
	Bindings []Value // values for each free variable in Fn.FreeVars
}

// The MakeMap instruction creates a new hash-table-based map object
// and yields a value of kind map.
//
// Type() returns a *types.Map.
//
// Example printed form:
// 	t1 = make map[string]int t0
//
type MakeMap struct {
	Register
	Reserve Value // initial space reservation; nil => default
}

// The MakeChan instruction creates a new channel object and yields a
// value of kind chan.
//
// Type() returns a *types.Chan.
//
// Example printed form:
// 	t0 = make chan int 0
//
type MakeChan struct {
	Register
	Size Value // int; size of buffer; zero => synchronous.
}

// MakeSlice yields a slice of length Len backed by a newly allocated
// array of length Cap.
//
// Both Len and Cap must be non-nil Values of integer type.
//
// (Alloc(types.Array) followed by Slice will not suffice because
// Alloc can only create arrays of statically known length.)
//
// Type() returns a *types.Slice.
//
// Example printed form:
// 	t1 = make slice []string 1:int t0
//
type MakeSlice struct {
	Register
	Len Value
	Cap Value
}

// Slice yields a slice of an existing string, slice or *array X
// between optional integer bounds Low and High.
//
// Type() returns string if the type of X was string, otherwise a
// *types.Slice with the same element type as X.
//
// Example printed form:
// 	t1 = slice t0[1:]
//
type Slice struct {
	Register
	X         Value // slice, string, or *array
	Low, High Value // either may be nil
}

// FieldAddr yields the address of Field of *struct  X.
//
// The field is identified by its index within the field list of the
// struct type of X.
//
// Type() returns a *types.Pointer.
//
// Example printed form:
// 	t1 = &t0.name [#1]
//
type FieldAddr struct {
	Register
	X     Value // *struct
	Field int   // index into X.Type().(*types.Struct).Fields
}

// Field yields the Field of struct X.
//
// The field is identified by its index within the field list of the
// struct type of X; by using numeric indices we avoid ambiguity of
// package-local identifiers and permit compact representations.
//
// Example printed form:
// 	t1 = t0.name [#1]
//
type Field struct {
	Register
	X     Value // struct
	Field int   // index into X.Type().(*types.Struct).Fields
}

// IndexAddr yields the address of the element at index Index of
// collection X.  Index is an integer expression.
//
// The elements of maps and strings are not addressable; use Lookup or
// MapUpdate instead.
//
// Type() returns a *types.Pointer.
//
// Example printed form:
// 	t2 = &t0[t1]
//
type IndexAddr struct {
	Register
	X     Value // slice or *array,
	Index Value // numeric index
}

// Index yields element Index of array X.
//
// TODO(adonovan): permit X to have type slice.
// Currently this requires IndexAddr followed by Load.
//
// Example printed form:
// 	t2 = t0[t1]
//
type Index struct {
	Register
	X     Value // array
	Index Value // integer index
}

// Lookup yields element Index of collection X, a map or string.
// Index is an integer expression if X is a string or the appropriate
// key type if X is a map.
//
// If CommaOk, the result is a 2-tuple of the value above and a
// boolean indicating the result of a map membership test for the key.
// The components of the tuple are accessed using Extract.
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

// Select tests whether (or blocks until) one or more of the specified
// sent or received states is entered.
//
// It returns a triple (index int, recv ?, recvOk bool) whose
// components, described below, must be accessed via the Extract
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
// received value; Otherwise it is unspecified.  recv has no useful
// type since it is conceptually the union of all possible received
// values.
//
// The third component of the triple, recvOk, is a boolean whose value
// is true iff the selected operation was a receive and the receive
// successfully yielded a value.
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

// Range yields an iterator over the domain and range of X.
// Elements are accessed via Next.
//
// Type() returns a *types.Result (tuple type).
//
// Example printed form:
// 	t0 = range "hello":string
//
type Range struct {
	Register
	X Value // array, *array, slice, string, map or chan
}

// Next reads and advances the iterator Iter and returns a 3-tuple
// value (ok, k, v).  If the iterator is not exhausted, ok is true and
// k and v are the next elements of the domain and range,
// respectively.  Otherwise ok is false and k and v are undefined.
//
// For channel iterators, k is the received value and v is always
// undefined.
//
// Components of the tuple are accessed using Extract.
//
// Type() returns a *types.Result (tuple type).
//
// Example printed form:
// 	t1 = next t0
//
type Next struct {
	Register
	Iter Value
}

// TypeAssert tests whether interface value X has type
// AssertedType.
//
// If CommaOk: on success it returns a pair (v, true) where v is a
// copy of value X; on failure it returns (z, false) where z is the
// zero value of that type.  The components of the pair must be
// accessed using the Extract instruction.
//
// If !CommaOk, on success it returns just the single value v; on
// failure it panics.
//
// Type() reflects the actual type of the result, possibly a pair
// (types.Result); AssertedType is the asserted type.
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

// Extract yields component Index of Tuple.
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

// Jump transfers control to the sole successor of its owning block.
//
// A Jump instruction must be the last instruction of its containing
// BasicBlock.
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
// Example printed form:
// 	if t0 goto done else body
//
type If struct {
	anInstruction
	Cond Value
}

// Ret returns values and control back to the calling function.
//
// len(Results) is always equal to the number of results in the
// function's signature.  A source-level 'return' statement with no
// operands in a multiple-return value function is desugared to make
// the results explicit.
//
// If len(Results) > 1, Ret returns a tuple value with the specified
// components which the caller must access using Extract instructions.
//
// There is no instruction to return a ready-made tuple like those
// returned by a "value,ok"-mode TypeAssert, Lookup or UnOp(ARROW) or
// a tail-call to a function with multiple result parameters.
// TODO(adonovan): consider defining one; but: dis- and re-assembling
// the tuple is unavoidable if assignability conversions are required
// on the components.
//
// Ret must be the last instruction of its containing BasicBlock.
// Such a block has no successors.
//
// Example printed form:
// 	ret
// 	ret nil:I, 2:int
//
type Ret struct {
	anInstruction
	Results []Value
}

// Go creates a new goroutine and calls the specified function
// within it.
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
	CallCommon
}

// Defer pushes the specified call onto a stack of functions
// to be called immediately prior to returning from the
// current function.
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
	CallCommon
}

// Send sends X on channel Chan.
//
// Example printed form:
// 	send t0 <- t1
//
type Send struct {
	anInstruction
	Chan, X Value
}

// Store stores Val at address Addr.
// Stores can be of arbitrary types.
//
// Example printed form:
// 	*x = y
//
type Store struct {
	anInstruction
	Addr Value
	Val  Value
}

// MapUpdate updates the association of Map[Key] to Value.
//
// Example printed form:
//	t0[t1] = t2
//
type MapUpdate struct {
	anInstruction
	Map   Value
	Key   Value
	Value Value
}

// Embeddable mix-ins used for common parts of other structs. --------------------

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
	num   int        // "name" of virtual register, e.g. "t0".  Not guaranteed unique.
	Type_ types.Type // type of virtual register
}

// AnInstruction is a mix-in embedded by all Instructions.
// It provides the implementations of the Block and SetBlock methods.
type anInstruction struct {
	Block_ *BasicBlock // the basic block of this instruction
}

// CallCommon is a mix-in embedded by Go, Defer and Call to hold the
// common parts of a function or method call.
//
// Each CallCommon exists in one of two modes, function call and
// interface method invocation, or "call" and "invoke" for short.
//
// 1. "call" mode: when Recv is nil, a CallCommon represents an
// ordinary function call of the value in Func.
//
// In the common case in which Func is a *Function, this indicates a
// statically dispatched call to a package-level function, an
// anonymous function, or a method of a named type.  Also statically
// dispatched, but less common, Func may be a *MakeClosure, indicating
// an immediately applied function literal with free variables.  Any
// other Value of Func indicates a dynamically dispatched function
// call.
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
// 2. "invoke" mode: when Recv is non-nil, a CallCommon represents a
// dynamically dispatched call to an interface method.  In this
// mode, Recv is the interface value and Method is the index of the
// method within the interface type of the receiver.
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
	Method      int       // index of interface method within Recv.Type().(*types.Interface).Methods
	Func        Value     // target of call, iff function call
	Args        []Value   // actual parameters, including receiver in invoke mode
	HasEllipsis bool      // true iff last Args is a slice  (needed?)
	Pos         token.Pos // position of call expression
}

func (v *Builtin) Type() types.Type { return v.Object.GetType() }
func (v *Builtin) Name() string     { return v.Object.GetName() }

func (v *Capture) Type() types.Type { return v.Outer.Type() }
func (v *Capture) Name() string     { return v.Outer.Name() }

func (v *Global) Type() types.Type { return v.Type_ }
func (v *Global) Name() string     { return v.Name_ }

func (v *Function) Name() string     { return v.Name_ }
func (v *Function) Type() types.Type { return v.Signature }

func (v *Parameter) Type() types.Type { return v.Type_ }
func (v *Parameter) Name() string     { return v.Name_ }

func (v *Alloc) Type() types.Type { return v.Type_ }
func (v *Alloc) Name() string     { return v.Name_ }

func (v *Register) Type() types.Type       { return v.Type_ }
func (v *Register) setType(typ types.Type) { v.Type_ = typ }
func (v *Register) Name() string           { return fmt.Sprintf("t%d", v.num) }
func (v *Register) setNum(num int)         { v.num = num }

func (v *anInstruction) Block() *BasicBlock         { return v.Block_ }
func (v *anInstruction) SetBlock(block *BasicBlock) { v.Block_ = block }

func (ms *Type) Type() types.Type { return ms.NamedType }
func (ms *Type) String() string   { return ms.Name() }
func (ms *Type) Name() string     { return ms.NamedType.Obj.Name }

func (p *Package) Name() string { return p.Types.Name }

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
func (p *Package) Const(name string) (l *Literal) {
	l, _ = p.Members[name].(*Literal)
	return
}

// Type returns the package-level type of the specified name,
// or nil if not found.
//
func (p *Package) Type(name string) (t *Type) {
	t, _ = p.Members[name].(*Type)
	return
}

// "Implements" relation boilerplate.
// Don't try to factor this using promotion and mix-ins: the long-hand
// form serves as better documentation, including in godoc.

func (*Alloc) ImplementsValue()           {}
func (*BinOp) ImplementsValue()           {}
func (*Builtin) ImplementsValue()         {}
func (*Call) ImplementsValue()            {}
func (*Capture) ImplementsValue()         {}
func (*ChangeInterface) ImplementsValue() {}
func (*Conv) ImplementsValue()            {}
func (*Extract) ImplementsValue()         {}
func (*Field) ImplementsValue()           {}
func (*FieldAddr) ImplementsValue()       {}
func (*Function) ImplementsValue()        {}
func (*Global) ImplementsValue()          {}
func (*Index) ImplementsValue()           {}
func (*IndexAddr) ImplementsValue()       {}
func (*Literal) ImplementsValue()         {}
func (*Lookup) ImplementsValue()          {}
func (*MakeChan) ImplementsValue()        {}
func (*MakeClosure) ImplementsValue()     {}
func (*MakeInterface) ImplementsValue()   {}
func (*MakeMap) ImplementsValue()         {}
func (*MakeSlice) ImplementsValue()       {}
func (*Next) ImplementsValue()            {}
func (*Parameter) ImplementsValue()       {}
func (*Phi) ImplementsValue()             {}
func (*Range) ImplementsValue()           {}
func (*Select) ImplementsValue()          {}
func (*Slice) ImplementsValue()           {}
func (*TypeAssert) ImplementsValue()      {}
func (*UnOp) ImplementsValue()            {}

func (*Function) ImplementsMember() {}
func (*Global) ImplementsMember()   {}
func (*Literal) ImplementsMember()  {}
func (*Type) ImplementsMember()     {}

func (*Alloc) ImplementsInstruction()           {}
func (*BinOp) ImplementsInstruction()           {}
func (*Call) ImplementsInstruction()            {}
func (*ChangeInterface) ImplementsInstruction() {}
func (*Conv) ImplementsInstruction()            {}
func (*Defer) ImplementsInstruction()           {}
func (*Extract) ImplementsInstruction()         {}
func (*Field) ImplementsInstruction()           {}
func (*FieldAddr) ImplementsInstruction()       {}
func (*Go) ImplementsInstruction()              {}
func (*If) ImplementsInstruction()              {}
func (*Index) ImplementsInstruction()           {}
func (*IndexAddr) ImplementsInstruction()       {}
func (*Jump) ImplementsInstruction()            {}
func (*Lookup) ImplementsInstruction()          {}
func (*MakeChan) ImplementsInstruction()        {}
func (*MakeClosure) ImplementsInstruction()     {}
func (*MakeInterface) ImplementsInstruction()   {}
func (*MakeMap) ImplementsInstruction()         {}
func (*MakeSlice) ImplementsInstruction()       {}
func (*MapUpdate) ImplementsInstruction()       {}
func (*Next) ImplementsInstruction()            {}
func (*Phi) ImplementsInstruction()             {}
func (*Range) ImplementsInstruction()           {}
func (*Ret) ImplementsInstruction()             {}
func (*Select) ImplementsInstruction()          {}
func (*Send) ImplementsInstruction()            {}
func (*Slice) ImplementsInstruction()           {}
func (*Store) ImplementsInstruction()           {}
func (*TypeAssert) ImplementsInstruction()      {}
func (*UnOp) ImplementsInstruction()            {}
