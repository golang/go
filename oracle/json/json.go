// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package json defines the oracle's JSON schema.
package json

// All 'pos' strings are of the form "file:line:col".
// TODO(adonovan): improve performance by sharing filename strings.
// TODO(adonovan): improve precision by providing the start/end
// interval when available.
//
// TODO(adonovan): consider richer encodings of types, functions,
// methods, etc.

// A Peers is the result of a 'peers' query.
// If Allocs is empty, the selected channel can't point to anything.
type Peers struct {
	Pos      string   `json:"pos"`                // location of the selected channel op (<-)
	Type     string   `json:"type"`               // type of the selected channel
	Allocs   []string `json:"allocs,omitempty"`   // locations of aliased make(chan) ops
	Sends    []string `json:"sends,omitempty"`    // locations of aliased ch<-x ops
	Receives []string `json:"receives,omitempty"` // locations of aliased <-ch ops
}

// A Referrers is the result of a 'referrers' query.
type Referrers struct {
	Pos    string   `json:"pos"`              // location of the query reference
	ObjPos string   `json:"objpos,omitempty"` // location of the definition
	Desc   string   `json:"desc"`             // description of the denoted object
	Refs   []string `json:"refs,omitempty"`   // locations of all references
}

type CalleesItem struct {
	Name string `json:"name"` // full name of called function
	Pos  string `json:"pos"`  // location of called function
}

// A Callees is the result of a 'callees' query.
//
// Callees is nonempty unless the call was a dynamic call on a
// provably nil func or interface value.
type Callees struct {
	Pos     string         `json:"pos"`               // location of selected call site
	Desc    string         `json:"desc"`              // description of call site
	Callees []*CalleesItem `json:"callees,omitempty"` // set of possible call targets
}

// A Caller is one element of the slice returned by a 'callers' query.
// (Callstack also contains a similar slice.)
//
// The root of the callgraph has an unspecified "Caller" string.
type Caller struct {
	Pos    string `json:"pos,omitempty"` // location of the calling function
	Desc   string `json:"desc"`          // description of call site
	Caller string `json:"caller"`        // full name of calling function
}

// A CallGraph is one element of the slice returned by a 'callgraph' query.
// The index of each item in the slice is used to identify it in the
// Callers adjacency list.
//
// Multiple nodes may have the same Name due to context-sensitive
// treatment of some functions.
//
// TODO(adonovan): perhaps include edge labels (i.e. callsites).
type CallGraph struct {
	Name     string `json:"name"`               // full name of function
	Pos      string `json:"pos"`                // location of function
	Children []int  `json:"children,omitempty"` // indices of child nodes in callgraph list
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

// An Implements is one element of the result of an 'implements' query.
// Each one indicates a row in the "implements" relation over
// package-level named types defined by the package containing the
// selection.
type Implements struct {
	I    string `json:"i"`    // full name of the interface type
	IPos string `json:"ipos"` // location of its definition
	C    string `json:"c"`    // full name of the concrete type
	CPos string `json:"cpos"` // location of its definition
}

// A DescribePTALabel describes a pointer analysis label.
//
// A "label" is an object that may be pointed to by a pointer, map,
// channel, 'func', slice or interface.  Labels include:
//    - functions
//    - globals
//    - arrays created by literals (e.g. []byte("foo")) and conversions ([]byte(s))
//    - stack- and heap-allocated variables (including composite literals)
//    - arrays allocated by append()
//    - channels, maps and arrays created by make()
//    - and their subelements, e.g. "alloc.y[*].z"
//
type DescribePTALabel struct {
	Pos  string `json:"pos"`  // location of syntax that allocated the object
	Desc string `json:"desc"` // description of the label
}

// A DescribePointer describes a single pointer: its type and the
// set of "labels" it points to.
//
type DescribePointer struct {
	Type    string             `json:"type"`              // (concrete) type of the pointer
	NamePos string             `json:"namepos,omitempty"` // location of type defn, if Named
	Labels  []DescribePTALabel `json:"labels,omitempty"`  // pointed-to objects
}

// A DescribeValue is the additional result of a 'describe' query
// if the selection indicates a value or expression.
//
// If the described value is an interface, it will have one PTS entry
// describing each concrete type that it may contain.  For each
// concrete type that is a pointer, the PTS entry describes the labels
// it may point to.  The same is true for reflect.Values, except the
// dynamic types needn't be concrete.
//
type DescribeValue struct {
	Type   string             `json:"type"`             // type of the expression
	Value  string             `json:"value,omitempty"`  // value of the expression, if constant
	ObjPos string             `json:"objpos,omitempty"` // location of the definition, if an Ident
	PTAErr string             `json:"ptaerr,omitempty"` // reason pointer analysis wasn't attempted
	PTS    []*DescribePointer `json:"pts,omitempty"`    // points-to set; an interface may have many
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

type PTAWarning struct {
	Pos     string `json:"pos"`     // location associated with warning
	Message string `json:"message"` // warning message
}

// A Result is the common result of any oracle query.
// It contains a query-specific result element.
//
// TODO(adonovan): perhaps include other info such as: analysis scope,
// raw query position, stack of ast nodes, query package, etc.
type Result struct {
	Mode string `json:"mode"` // mode of the query

	// Exactly one of the following fields is populated:
	// the one specified by 'mode'.
	Callees    *Callees      `json:"callees,omitempty"`
	Callers    []Caller      `json:"callers,omitempty"`
	Callgraph  []CallGraph   `json:"callgraph,omitempty"`
	Callstack  *CallStack    `json:"callstack,omitempty"`
	Describe   *Describe     `json:"describe,omitempty"`
	Freevars   []*FreeVar    `json:"freevars,omitempty"`
	Implements []*Implements `json:"implements,omitempty"`
	Peers      *Peers        `json:"peers,omitempty"`
	Referrers  *Referrers    `json:"referrers,omitempty"`

	Warnings []PTAWarning `json:"warnings,omitempty"` // warnings from pointer analysis
}
