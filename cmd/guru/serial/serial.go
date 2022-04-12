// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package serial defines the guru's schema for -json output.
//
// The output of a guru query is a stream of one or more JSON objects.
// This table shows the types of objects in the result stream for each
// query type.
//
//	Query      Result stream
//	-----      -------------
//	callees    Callees
//	callers    Caller ...
//	callstack  CallStack
//	definition Definition
//	describe   Describe
//	freevars   FreeVar ...
//	implements Implements
//	peers      Peers
//	pointsto   PointsTo ...
//	referrers  ReferrersInitial ReferrersPackage ...
//	what       What
//	whicherrs  WhichErrs
//
// All 'pos' strings in the output are of the form "file:line:col",
// where line is the 1-based line number and col is the 1-based byte index.
package serial

// A Peers is the result of a 'peers' query.
// If Allocs is empty, the selected channel can't point to anything.
type Peers struct {
	Pos      string   `json:"pos"`                // location of the selected channel op (<-)
	Type     string   `json:"type"`               // type of the selected channel
	Allocs   []string `json:"allocs,omitempty"`   // locations of aliased make(chan) ops
	Sends    []string `json:"sends,omitempty"`    // locations of aliased ch<-x ops
	Receives []string `json:"receives,omitempty"` // locations of aliased <-ch ops
	Closes   []string `json:"closes,omitempty"`   // locations of aliased close(ch) ops
}

// A "referrers" query emits a ReferrersInitial object followed by zero or
// more ReferrersPackage objects, one per package that contains a reference.
type (
	ReferrersInitial struct {
		ObjPos string `json:"objpos,omitempty"` // location of the definition
		Desc   string `json:"desc"`             // description of the denoted object
	}
	ReferrersPackage struct {
		Package string `json:"package"`
		Refs    []Ref  `json:"refs"` // non-empty list of references within this package
	}
	Ref struct {
		Pos  string `json:"pos"`  // location of all references
		Text string `json:"text"` // text of the referring line
	}
)

// A Definition is the result of a 'definition' query.
type Definition struct {
	ObjPos string `json:"objpos,omitempty"` // location of the definition
	Desc   string `json:"desc"`             // description of the denoted object
}

// A Callees is the result of a 'callees' query.
//
// Callees is nonempty unless the call was a dynamic call on a
// provably nil func or interface value.
type (
	Callees struct {
		Pos     string    `json:"pos"`  // location of selected call site
		Desc    string    `json:"desc"` // description of call site
		Callees []*Callee `json:"callees"`
	}
	Callee struct {
		Name string `json:"name"` // full name of called function
		Pos  string `json:"pos"`  // location of called function
	}
)

// A Caller is one element of the slice returned by a 'callers' query.
// (Callstack also contains a similar slice.)
//
// The root of the callgraph has an unspecified "Caller" string.
type Caller struct {
	Pos    string `json:"pos,omitempty"` // location of the calling function
	Desc   string `json:"desc"`          // description of call site
	Caller string `json:"caller"`        // full name of calling function
}

// A CallStack is the result of a 'callstack' query.
// It indicates an arbitrary path from the root of the callgraph to
// the query function.
//
// If the Callers slice is empty, the function was unreachable in this
// analysis scope.
type CallStack struct {
	Pos     string   `json:"pos"`     // location of the selected function
	Target  string   `json:"target"`  // the selected function
	Callers []Caller `json:"callers"` // enclosing calls, innermost first.
}

// A FreeVar is one element of the slice returned by a 'freevars'
// query.  Each one identifies an expression referencing a local
// identifier defined outside the selected region.
type FreeVar struct {
	Pos  string `json:"pos"`  // location of the identifier's definition
	Kind string `json:"kind"` // one of {var,func,type,const,label}
	Ref  string `json:"ref"`  // referring expression (e.g. "x" or "x.y.z")
	Type string `json:"type"` // type of the expression
}

// An Implements contains the result of an 'implements' query.
// It describes the queried type, the set of named non-empty interface
// types to which it is assignable, and the set of named/*named types
// (concrete or non-empty interface) which may be assigned to it.
type Implements struct {
	T                 ImplementsType   `json:"type,omitempty"`    // the queried type
	AssignableTo      []ImplementsType `json:"to,omitempty"`      // types assignable to T
	AssignableFrom    []ImplementsType `json:"from,omitempty"`    // interface types assignable from T
	AssignableFromPtr []ImplementsType `json:"fromptr,omitempty"` // interface types assignable only from *T

	// The following fields are set only if the query was a method.
	// Assignable{To,From,FromPtr}Method[i] is the corresponding
	// method of type Assignable{To,From,FromPtr}[i], or blank
	// {"",""} if that type lacks the method.
	Method                  *DescribeMethod  `json:"method,omitempty"` //  the queried method
	AssignableToMethod      []DescribeMethod `json:"to_method,omitempty"`
	AssignableFromMethod    []DescribeMethod `json:"from_method,omitempty"`
	AssignableFromPtrMethod []DescribeMethod `json:"fromptr_method,omitempty"`
}

// An ImplementsType describes a single type as part of an 'implements' query.
type ImplementsType struct {
	Name string `json:"name"` // full name of the type
	Pos  string `json:"pos"`  // location of its definition
	Kind string `json:"kind"` // "basic", "array", etc
}

// A SyntaxNode is one element of a stack of enclosing syntax nodes in
// a "what" query.
type SyntaxNode struct {
	Description string `json:"desc"`  // description of syntax tree
	Start       int    `json:"start"` // start byte offset, 0-based
	End         int    `json:"end"`   // end byte offset
}

// A What is the result of the "what" query, which quickly identifies
// the selection, parsing only a single file.  It is intended for use
// in low-latency GUIs.
type What struct {
	Enclosing  []SyntaxNode `json:"enclosing"`            // enclosing nodes of syntax tree
	Modes      []string     `json:"modes"`                // query modes enabled for this selection.
	SrcDir     string       `json:"srcdir,omitempty"`     // $GOROOT src directory containing queried package
	ImportPath string       `json:"importpath,omitempty"` // import path of queried package
	Object     string       `json:"object,omitempty"`     // name of identified object, if any
	SameIDs    []string     `json:"sameids,omitempty"`    // locations of references to same object
}

// A PointsToLabel describes a pointer analysis label.
//
// A "label" is an object that may be pointed to by a pointer, map,
// channel, 'func', slice or interface.  Labels include:
//   - functions
//   - globals
//   - arrays created by literals (e.g. []byte("foo")) and conversions ([]byte(s))
//   - stack- and heap-allocated variables (including composite literals)
//   - arrays allocated by append()
//   - channels, maps and arrays created by make()
//   - and their subelements, e.g. "alloc.y[*].z"
type PointsToLabel struct {
	Pos  string `json:"pos"`  // location of syntax that allocated the object
	Desc string `json:"desc"` // description of the label
}

// A PointsTo is one element of the result of a 'pointsto' query on an
// expression.  It describes a single pointer: its type and the set of
// "labels" it points to.
//
// If the pointer is of interface type, it will have one PTS entry
// describing each concrete type that it may contain.  For each
// concrete type that is a pointer, the PTS entry describes the labels
// it may point to.  The same is true for reflect.Values, except the
// dynamic types needn't be concrete.
type PointsTo struct {
	Type    string          `json:"type"`              // (concrete) type of the pointer
	NamePos string          `json:"namepos,omitempty"` // location of type defn, if Named
	Labels  []PointsToLabel `json:"labels,omitempty"`  // pointed-to objects
}

// A DescribeValue is the additional result of a 'describe' query
// if the selection indicates a value or expression.
type DescribeValue struct {
	Type     string       `json:"type"`               // type of the expression
	Value    string       `json:"value,omitempty"`    // value of the expression, if constant
	ObjPos   string       `json:"objpos,omitempty"`   // location of the definition, if an Ident
	TypesPos []Definition `json:"typespos,omitempty"` // location of the named types, that type consist of
}

type DescribeMethod struct {
	Name string `json:"name"` // method name, as defined by types.Selection.String()
	Pos  string `json:"pos"`  // location of the method's definition
}

// A DescribeType is the additional result of a 'describe' query
// if the selection indicates a type.
type DescribeType struct {
	Type    string           `json:"type"`              // the string form of the type
	NamePos string           `json:"namepos,omitempty"` // location of definition of type, if named
	NameDef string           `json:"namedef,omitempty"` // underlying definition of type, if named
	Methods []DescribeMethod `json:"methods,omitempty"` // methods of the type
}

type DescribeMember struct {
	Name    string           `json:"name"`              // name of member
	Type    string           `json:"type,omitempty"`    // type of member (underlying, if 'type')
	Value   string           `json:"value,omitempty"`   // value of member (if 'const')
	Pos     string           `json:"pos"`               // location of definition of member
	Kind    string           `json:"kind"`              // one of {var,const,func,type}
	Methods []DescribeMethod `json:"methods,omitempty"` // methods (if member is a type)
}

// A DescribePackage is the additional result of a 'describe' if
// the selection indicates a package.
type DescribePackage struct {
	Path    string            `json:"path"`              // import path of the package
	Members []*DescribeMember `json:"members,omitempty"` // accessible members of the package
}

// A Describe is the result of a 'describe' query.
// It may contain an element describing the selected semantic entity
// in detail.
type Describe struct {
	Desc   string `json:"desc"`             // description of the selected syntax node
	Pos    string `json:"pos"`              // location of the selected syntax node
	Detail string `json:"detail,omitempty"` // one of {package, type, value}, or "".

	// At most one of the following fields is populated:
	// the one specified by 'detail'.
	Package *DescribePackage `json:"package,omitempty"`
	Type    *DescribeType    `json:"type,omitempty"`
	Value   *DescribeValue   `json:"value,omitempty"`
}

// A WhichErrs is the result of a 'whicherrs' query.
// It contains the position of the queried error and the possible globals,
// constants, and types it may point to.
type WhichErrs struct {
	ErrPos    string          `json:"errpos,omitempty"`    // location of queried error
	Globals   []string        `json:"globals,omitempty"`   // locations of globals
	Constants []string        `json:"constants,omitempty"` // locations of constants
	Types     []WhichErrsType `json:"types,omitempty"`     // Types
}

type WhichErrsType struct {
	Type     string `json:"type,omitempty"`
	Position string `json:"position,omitempty"`
}
