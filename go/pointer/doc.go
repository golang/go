// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Package pointer implements Andersen's analysis, an inclusion-based
pointer analysis algorithm first described in (Andersen, 1994).

The implementation is similar to that described in (Pearce et al,
PASTE'04).  Unlike many algorithms which interleave constraint
generation and solving, constructing the callgraph as they go, this
implementation for the most part observes a phase ordering (generation
before solving), with only simple (copy) constraints being generated
during solving.  (The exception is reflection, which creates various
constraints during solving as new types flow to reflect.Value
operations.)  This improves the traction of presolver optimisations,
but imposes certain restrictions, e.g. potential context sensitivity
is limited since all variants must be created a priori.

We intend to add various presolving optimisations such as Pointer and
Location Equivalence from (Hardekopf & Lin, SAS '07) and solver
optimisatisions such as Hybrid- and Lazy- Cycle Detection from
(Hardekopf & Lin, PLDI'07),


CLASSIFICATION

Our algorithm is INCLUSION-BASED: the points-to sets for x and y will
be related by pts(y) ⊇ pts(x) if the program contains the statement
y = x.

It is FLOW-INSENSITIVE: it ignores all control flow constructs and the
order of statements in a program.  It is therefore a "MAY ALIAS"
analysis: its facts are of the form "P may/may not point to L",
not "P must point to L".

It is FIELD-SENSITIVE: it builds separate points-to sets for distinct
fields, such as x and y in struct { x, y *int }.

It is mostly CONTEXT-INSENSITIVE: most functions are analyzed once,
so values can flow in at one call to the function and return out at
another.  Only some smaller functions are analyzed with consideration
to their calling context.

It has a CONTEXT-SENSITIVE HEAP: objects are named by both allocation
site and context, so the objects returned by two distinct calls to f:
   func f() *T { return new(T) }
are distinguished up to the limits of the calling context.

It is a WHOLE PROGRAM analysis: it requires SSA-form IR for the
complete Go program and summaries for native code.

See the (Hind, PASTE'01) survey paper for an explanation of these terms.


TERMINOLOGY

We occasionally use C's x->f notation to distinguish the case where x
is a struct pointer from x.f where is a struct value.


NODES

Nodes are the key datastructure of the analysis, and have a dual role:
they represent both constraint variables (equivalence classes of
pointers) and members of points-to sets (things that can be pointed
at, i.e. "labels").

Nodes are naturally numbered.  The numbering enables compact
representations of sets of nodes such as bitvectors or BDDs; and the
ordering enables a very cheap way to group related nodes together.
(For example, passing n parameters consists of generating n parallel
constraints from caller+i to callee+i for 0<=i<n.)

The zero nodeid means "not a pointer".  Currently it's only used for
struct{} or ().  We generate all flow constraints, even for non-pointer
types, with the expectations that (a) presolver optimisations will
quickly collapse all the non-pointer ones and (b) we may get more
precise results by treating uintptr as a possible pointer.

Each node represents a scalar (word-sized) part of a value or object.
Aggregate types (structs, tuples, arrays) are recursively flattened
out into a sequential list of scalar component types, and all the
elements of an array are represented by a single node.  (The
flattening of a basic type is a list containing a single node.)

Nodes are connected into a graph with various kinds of labelled edges:
simple edges (or copy constraints) represent value flow.  Complex
edges (load, store, etc) trigger the creation of new simple edges
during the solving phase.


OBJECTS

Conceptually, an "object" is a contiguous sequence of nodes denoting
an addressable location: something that a pointer can point to.  The
first node of an object has a non-nil obj field containing information
about the allocation: its size, context, and ssa.Value.

Objects include:
   - functions and globals;
   - variable allocations in the stack frame or heap;
   - maps, channels and slices created by calls to make();
   - allocations to construct an interface;
   - allocations caused by literals and conversions,
     e.g. []byte("foo"), []byte(str).
   - arrays allocated by calls to append();

Many objects have no Go types.  For example, the func, map and chan
type kinds in Go are all varieties of pointers, but the respective
objects are actual functions, maps, and channels.  Given the way we
model interfaces, they too are pointers to tagged objects with no
Go type.  And an *ssa.Global denotes the address of a global variable,
but the object for a Global is the actual data.  So, types of objects
are usually "off by one indirection".

The individual nodes of an object are sometimes referred to as
"labels".

Objects containing no nodes (i.e. just empty structs; tuples may be
values but never objects in Go) are padded with an invalid-type node
to have at least one node so that there is something to point to.
(TODO(adonovan): I think this is unnecessary now that we have identity
nodes; check.)


TAGGED OBJECTS

An tagged object has the following layout:

    T          -- obj.flags ⊇ {otTagged}
    v
    ...

The T node's typ field is the dynamic type of the "payload", the value
v which follows, flattened out.  The T node's obj has the otTagged
flag.

Tagged objects are needed when generalizing across types: interfaces,
reflect.Values, reflect.Types.  Each of these three types is modelled
as a pointer that exclusively points to tagged objects.

Tagged objects may be indirect (obj.flags ⊇ {otIndirect}) meaning that
the value v is not of type T but *T; this is used only for
reflect.Values that represent lvalues.


ANALYSIS ABSTRACTION OF EACH TYPE

Variables of the following "scalar" types may be represented by a
single node: basic types, pointers, channels, maps, slices, 'func'
pointers, interfaces.

Pointers
  Nothing to say here.

Basic types (bool, string, numbers, unsafe.Pointer)
  Currently all fields in the flattening of a type, including
  non-pointer basic types such as int, are represented in objects and
  values.  Though non-pointer nodes within values are uninteresting,
  non-pointer nodes in objects may be useful (if address-taken)
  because they permit the analysis to deduce, in this example,

     var s struct{ ...; x int; ... }
     p := &s.x

  that p points to s.x.  If we ignored such object fields, we could only
  say that p points somewhere within s.

  All other basic types are ignored.  Expressions of these types have
  zero nodeid, and fields of these types within aggregate other types
  are omitted.

  unsafe.Pointer conversions are not yet modelled as pointer
  conversions.  Consequently uintptr is always a number and uintptr
  nodes do not point to any object.

Channels
  An expression of type 'chan T' is a kind of pointer that points
  exclusively to channel objects, i.e. objects created by MakeChan (or
  reflection).

  'chan T' is treated like *T.
  *ssa.MakeChan is treated as equivalent to new(T).
  *ssa.Send and receive (*ssa.UnOp(ARROW)) and are equivalent to store
   and load.

Maps
  An expression of type 'map[K]V' is a kind of pointer that points
  exclusively to map objects, i.e. objects created by MakeMap (or
  reflection).

  map K[V] is treated like *M where M = struct{k K; v V}.
  *ssa.MakeMap is equivalent to new(M).
  *ssa.MapUpdate is equivalent to *y=x where *y and x have type M.
  *ssa.Lookup is equivalent to y=x.v where x has type *M.

Slices
  A slice []T, which dynamically resembles a struct{array *T, len, cap int},
  is treated as if it were just a *T pointer; the len and cap fields are
  ignored.

  *ssa.MakeSlice is treated like new([1]T): an allocation of a
   singleton array.
  *ssa.Index on a slice is equivalent to a load.
  *ssa.IndexAddr on a slice returns the address of the sole element of the
  slice, i.e. the same address.
  *ssa.Slice is treated as a simple copy.

Functions
  An expression of type 'func...' is a kind of pointer that points
  exclusively to function objects.

  A function object has the following layout:

     identity         -- typ:*types.Signature; obj.flags ⊇ {otFunction}
     params_0         -- (the receiver, if a method)
     ...
     params_n-1
     results_0
     ...
     results_m-1

  There may be multiple function objects for the same *ssa.Function
  due to context-sensitive treatment of some functions.

  The first node is the function's identity node.
  Associated with every callsite is a special "targets" variable,
  whose pts(·) contains the identity node of each function to which
  the call may dispatch.  Identity words are not otherwise used.

  The following block of nodes represent the flattened-out types of
  the parameters and results of the function object, and are
  collectively known as its "P/R block".

  The treatment of free variables of closures (*ssa.Capture) is like
  that of global variables; it is not context-sensitive.
  *ssa.MakeClosure instructions create copy edges to Captures.

  A Go value of type 'func' (i.e. a pointer to one or more functions)
  is a pointer whose pts() contains function objects.  The valueNode()
  for an *ssa.Function returns a singleton for that function.

Interfaces
  An expression of type 'interface{...}' is a kind of pointer that
  points exclusively to tagged objects.  All tagged objects pointed to
  by an interface are direct (the otIndirect flag is clear) and
  concrete (the tag type T is not itself an interface type).  The
  associated ssa.Value for an interface's tagged objects may be an
  *ssa.MakeInterface instruction, or nil if the tagged object was
  created by an instrinsic (e.g. reflection).

  Constructing an interface value causes generation of constraints for
  all of the concrete type's methods; we can't tell a priori which
  ones may be called.

  TypeAssert y = x.(T) is implemented by a dynamic filter triggered by
  each tagged object E added to pts(x).  If T is an interface that E.T
  implements, E is added to pts(y).  If T is a concrete type then edge
  E.v -> pts(y) is added.

  ChangeInterface is a simple copy because the representation of
  tagged objects is independent of the interface type (in contrast
  to the "method tables" approach used by the gc runtime).

  y := Invoke x.m(...) is implemented by allocating a contiguous P/R
  block for the callsite and adding a dynamic rule triggered by each
  tagged object E added to pts(x).  The rule adds param/results copy
  edges to/from each discovered concrete method.

  (Q. Why do we model an interface as a pointer to a pair of type and
  value, rather than as a pair of a pointer to type and a pointer to
  value?
  A. Control-flow joins would merge interfaces ({T1}, {V1}) and ({T2},
  {V2}) to make ({T1,T2}, {V1,V2}), leading to the infeasible and
  type-unsafe combination (T1,V2).  Treating the value and its concrete
  type as inseparable makes the analysis type-safe.)

reflect.Value
  A reflect.Value is modelled very similar to an interface{}, i.e. as
  a pointer exclusively to tagged objects, but with two
  generalizations.

  1) a reflect.Value that represents an lvalue points to an indirect
     (obj.flags ⊇ {otIndirect}) tagged object, which has a similar
     layout to an tagged object except that the value is a pointer to
     the dynamic type.  Indirect tagged objects preserve the correct
     aliasing so that mutations made by (reflect.Value).Set can be
     observed.

     Indirect objects only arise when an lvalue is derived from an
     rvalue by indirection, e.g. the following code:

        type S struct { X T }
        var s S
        var i interface{} = &s    // i points to a *S-tagged object (from MakeInterface)
        v1 := reflect.ValueOf(i)  // v1 points to same *S-tagged object as i
        v2 := v1.Elem()           // v2 points to an indirect S-tagged object, pointing to s
        v3 := v2.FieldByName("X") // v3 points to an indirect int-tagged object, pointing to s.X
        v3.Set(y)                 // pts(s.X) ⊇ pts(y)

     Whether indirect or not, the concrete type of the tagged object
     corresponds to the user-visible dynamic type, and the existence
     of a pointer is an implementation detail.

  2) The dynamic type tag of a tagged object pointed to by a
     reflect.Value may be an interface type; it need not be concrete.

     This arises in code such as this:
        tEface := reflect.TypeOf(new(interface{}).Elem() // interface{}
        eface := reflect.Zero(tEface)
     pts(eface) is a singleton containing an interface{}-tagged
     object.  That tagged object's payload is an interface{} value,
     i.e. the pts of the payload contains only concrete-tagged
     objects, although in this example it's the zero interface{} value,
     so its pts is empty.

reflect.Type
  Just as in the real "reflect" library, we represent a reflect.Type
  as an interface whose sole implementation is the concrete type,
  *reflect.rtype.  (This choice is forced on us by go/types: clients
  cannot fabricate types with arbitrary method sets.)

  rtype instances are canonical: there is at most one per dynamic
  type.  (rtypes are in fact large structs but since identity is all
  that matters, we represent them by a single node.)

  The payload of each *rtype-tagged object is an *rtype pointer that
  points to exactly one such canonical rtype object.  We exploit this
  by setting the node.typ of the payload to the dynamic type, not
  '*rtype'.  This saves us an indirection in each resolution rule.  As
  an optimisation, *rtype-tagged objects are canonicalized too.


Aggregate types:

Aggregate types are treated as if all directly contained
aggregates are recursively flattened out.

Structs
  *ssa.Field y = x.f creates a simple edge to y from x's node at f's offset.

  *ssa.FieldAddr y = &x->f requires a dynamic closure rule to create
   simple edges for each struct discovered in pts(x).

  The nodes of a struct consist of a special 'identity' node (whose
  type is that of the struct itself), followed by the nodes for all
  the struct's fields, recursively flattened out.  A pointer to the
  struct is a pointer to its identity node.  That node allows us to
  distinguish a pointer to a struct from a pointer to its first field.

  Field offsets are currently the logical field offsets (plus one for
  the identity node), so the sizes of the fields can be ignored by the
  analysis.

  Sound treatment of unsafe.Pointer conversions (not yet implemented)
  would require us to model memory layout using physical field offsets
  to ascertain which object field(s) might be aliased by a given
  FieldAddr of a different base pointer type.  It would also require
  us to dispense with the identity node.

  *ssa.Field y = x.f creates a simple edge to y from x's node at f's offset.

  *ssa.FieldAddr y = &x->f requires a dynamic closure rule to create
   simple edges for each struct discovered in pts(x).

Arrays
  We model an array by an identity node (whose type is that of the
  array itself) followed by a node representing all the elements of
  the array; the analysis does not distinguish elements with different
  indices.  Effectively, an array is treated like struct{elem T}, a
  load y=x[i] like y=x.elem, and a store x[i]=y like x.elem=y; the
  index i is ignored.

  A pointer to an array is pointer to its identity node.  (A slice is
  also a pointer to an array's identity node.)  The identity node
  allows us to distinguish a pointer to an array from a pointer to one
  of its elements, but it is rather costly because it introduces more
  offset constraints into the system.  Furthermore, sound treatment of
  unsafe.Pointer would require us to dispense with this node.

  Arrays may be allocated by Alloc, by make([]T), by calls to append,
  and via reflection.

Tuples (T, ...)
  Tuples are treated like structs with naturally numbered fields.
  *ssa.Extract is analogous to *ssa.Field.

  However, tuples have no identity field since by construction, they
  cannot be address-taken.


FUNCTION CALLS

  There are three kinds of function call:
  (1) static "call"-mode calls of functions.
  (2) dynamic "call"-mode calls of functions.
  (3) dynamic "invoke"-mode calls of interface methods.
  Cases 1 and 2 apply equally to methods and standalone functions.

  Static calls.
    A static call consists three steps:
    - finding the function object of the callee;
    - creating copy edges from the actual parameter value nodes to the
      params block in the function object (this includes the receiver
      if the callee is a method);
    - creating copy edges from the results block in the function
      object to the value nodes for the result of the call.

    Context sensitivity

      Static calls (alone) may be treated context sensitively,
      i.e. each callsite may cause a distinct re-analysis of the
      callee, improving precision.  Our current context-sensitivity
      policy treats all intrinsics and getter/setter methods in this
      manner since such functions are small and seem like an obvious
      source of spurious confluences, though this has not yet been
      evaluated.

  Dynamic function calls

    Dynamic calls work in a similar manner except that the creation of
    copy edges occurs dynamically, in a similar fashion to a pair of
    struct copies:

      *fn->params = callargs
      callresult = *fn->results

    (Recall that the function object's params and results blocks are
    contiguous.)

  Interface method invocation

    For invoke-mode calls, we create a params/results block for the
    callsite and attach a dynamic closure rule to the interface.  For
    each new tagged object that flows to the interface, we look up
    the concrete method, find its function object, and connect its P/R
    block to the callsite's P/R block.

  Recording call targets

    The analysis notifies its clients of each callsite it encounters,
    passing a CallSite interface.  Among other things, the CallSite
    contains a synthetic constraint variable ("targets") whose
    points-to solution includes the set of all function objects to
    which the call may dispatch.

    It is via this mechanism that the callgraph is made available.
    Clients may also elect to be notified of callgraph edges directly;
    internally this just iterates all "targets" variables' pts(·)s.


SOLVER

The solver is currently a very naive Andersen-style implementation,
although it does use difference propagation (Pearce et al, SQC'04).
There is much to do.


FURTHER READING.

Andersen, L. O. 1994. Program analysis and specialization for the C
programming language. Ph.D. dissertation. DIKU, University of
Copenhagen.

David J. Pearce, Paul H. J. Kelly, and Chris Hankin. 2004.  Efficient
field-sensitive pointer analysis for C. In Proceedings of the 5th ACM
SIGPLAN-SIGSOFT workshop on Program analysis for software tools and
engineering (PASTE '04). ACM, New York, NY, USA, 37-42.
http://doi.acm.org/10.1145/996821.996835

David J. Pearce, Paul H. J. Kelly, and Chris Hankin. 2004. Online
Cycle Detection and Difference Propagation: Applications to Pointer
Analysis. Software Quality Control 12, 4 (December 2004), 311-337.
http://dx.doi.org/10.1023/B:SQJO.0000039791.93071.a2

David Grove and Craig Chambers. 2001. A framework for call graph
construction algorithms. ACM Trans. Program. Lang. Syst. 23, 6
(November 2001), 685-746.
http://doi.acm.org/10.1145/506315.506316

Ben Hardekopf and Calvin Lin. 2007. The ant and the grasshopper: fast
and accurate pointer analysis for millions of lines of code. In
Proceedings of the 2007 ACM SIGPLAN conference on Programming language
design and implementation (PLDI '07). ACM, New York, NY, USA, 290-299.
http://doi.acm.org/10.1145/1250734.1250767

Ben Hardekopf and Calvin Lin. 2007. Exploiting pointer and location
equivalence to optimize pointer analysis. In Proceedings of the 14th
international conference on Static Analysis (SAS'07), Hanne Riis
Nielson and Gilberto Filé (Eds.). Springer-Verlag, Berlin, Heidelberg,
265-280.

*/
package pointer
