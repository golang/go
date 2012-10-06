// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parse nodes.

package parse

import (
	"bytes"
	"fmt"
	"strconv"
	"strings"
)

// A Node is an element in the parse tree. The interface is trivial.
// The interface contains an unexported method so that only
// types local to this package can satisfy it.
type Node interface {
	Type() NodeType
	String() string
	// Copy does a deep copy of the Node and all its components.
	// To avoid type assertions, some XxxNodes also have specialized
	// CopyXxx methods that return *XxxNode.
	Copy() Node
	Position() Pos // byte position of start of node in full original input string
	// Make sure only functions in this package can create Nodes.
	unexported()
}

// NodeType identifies the type of a parse tree node.
type NodeType int

// Pos represents a byte position in the original input text from which
// this template was parsed.
type Pos int

func (p Pos) Position() Pos {
	return p
}

// unexported keeps Node implementations local to the package.
// All implementations embed Pos, so this takes care of it.
func (Pos) unexported() {
}

// Type returns itself and provides an easy default implementation
// for embedding in a Node. Embedded in all non-trivial Nodes.
func (t NodeType) Type() NodeType {
	return t
}

const (
	NodeText       NodeType = iota // Plain text.
	NodeAction                     // A non-control action such as a field evaluation.
	NodeBool                       // A boolean constant.
	NodeChain                      // A sequence of field accesses.
	NodeCommand                    // An element of a pipeline.
	NodeDot                        // The cursor, dot.
	nodeElse                       // An else action. Not added to tree.
	nodeEnd                        // An end action. Not added to tree.
	NodeField                      // A field or method name.
	NodeIdentifier                 // An identifier; always a function name.
	NodeIf                         // An if action.
	NodeList                       // A list of Nodes.
	NodeNil                        // An untyped nil constant.
	NodeNumber                     // A numerical constant.
	NodePipe                       // A pipeline of commands.
	NodeRange                      // A range action.
	NodeString                     // A string constant.
	NodeTemplate                   // A template invocation action.
	NodeVariable                   // A $ variable.
	NodeWith                       // A with action.
)

// Nodes.

// ListNode holds a sequence of nodes.
type ListNode struct {
	NodeType
	Pos
	Nodes []Node // The element nodes in lexical order.
}

func newList(pos Pos) *ListNode {
	return &ListNode{NodeType: NodeList, Pos: pos}
}

func (l *ListNode) append(n Node) {
	l.Nodes = append(l.Nodes, n)
}

func (l *ListNode) String() string {
	b := new(bytes.Buffer)
	for _, n := range l.Nodes {
		fmt.Fprint(b, n)
	}
	return b.String()
}

func (l *ListNode) CopyList() *ListNode {
	if l == nil {
		return l
	}
	n := newList(l.Pos)
	for _, elem := range l.Nodes {
		n.append(elem.Copy())
	}
	return n
}

func (l *ListNode) Copy() Node {
	return l.CopyList()
}

// TextNode holds plain text.
type TextNode struct {
	NodeType
	Pos
	Text []byte // The text; may span newlines.
}

func newText(pos Pos, text string) *TextNode {
	return &TextNode{NodeType: NodeText, Pos: pos, Text: []byte(text)}
}

func (t *TextNode) String() string {
	return fmt.Sprintf("%q", t.Text)
}

func (t *TextNode) Copy() Node {
	return &TextNode{NodeType: NodeText, Text: append([]byte{}, t.Text...)}
}

// PipeNode holds a pipeline with optional declaration
type PipeNode struct {
	NodeType
	Pos
	Line int             // The line number in the input (deprecated; kept for compatibility)
	Decl []*VariableNode // Variable declarations in lexical order.
	Cmds []*CommandNode  // The commands in lexical order.
}

func newPipeline(pos Pos, line int, decl []*VariableNode) *PipeNode {
	return &PipeNode{NodeType: NodePipe, Pos: pos, Line: line, Decl: decl}
}

func (p *PipeNode) append(command *CommandNode) {
	p.Cmds = append(p.Cmds, command)
}

func (p *PipeNode) String() string {
	s := ""
	if len(p.Decl) > 0 {
		for i, v := range p.Decl {
			if i > 0 {
				s += ", "
			}
			s += v.String()
		}
		s += " := "
	}
	for i, c := range p.Cmds {
		if i > 0 {
			s += " | "
		}
		s += c.String()
	}
	return s
}

func (p *PipeNode) CopyPipe() *PipeNode {
	if p == nil {
		return p
	}
	var decl []*VariableNode
	for _, d := range p.Decl {
		decl = append(decl, d.Copy().(*VariableNode))
	}
	n := newPipeline(p.Pos, p.Line, decl)
	for _, c := range p.Cmds {
		n.append(c.Copy().(*CommandNode))
	}
	return n
}

func (p *PipeNode) Copy() Node {
	return p.CopyPipe()
}

// ActionNode holds an action (something bounded by delimiters).
// Control actions have their own nodes; ActionNode represents simple
// ones such as field evaluations and parenthesized pipelines.
type ActionNode struct {
	NodeType
	Pos
	Line int       // The line number in the input (deprecated; kept for compatibility)
	Pipe *PipeNode // The pipeline in the action.
}

func newAction(pos Pos, line int, pipe *PipeNode) *ActionNode {
	return &ActionNode{NodeType: NodeAction, Pos: pos, Line: line, Pipe: pipe}
}

func (a *ActionNode) String() string {
	return fmt.Sprintf("{{%s}}", a.Pipe)

}

func (a *ActionNode) Copy() Node {
	return newAction(a.Pos, a.Line, a.Pipe.CopyPipe())

}

// CommandNode holds a command (a pipeline inside an evaluating action).
type CommandNode struct {
	NodeType
	Pos
	Args []Node // Arguments in lexical order: Identifier, field, or constant.
}

func newCommand(pos Pos) *CommandNode {
	return &CommandNode{NodeType: NodeCommand, Pos: pos}
}

func (c *CommandNode) append(arg Node) {
	c.Args = append(c.Args, arg)
}

func (c *CommandNode) String() string {
	s := ""
	for i, arg := range c.Args {
		if i > 0 {
			s += " "
		}
		if arg, ok := arg.(*PipeNode); ok {
			s += "(" + arg.String() + ")"
			continue
		}
		s += arg.String()
	}
	return s
}

func (c *CommandNode) Copy() Node {
	if c == nil {
		return c
	}
	n := newCommand(c.Pos)
	for _, c := range c.Args {
		n.append(c.Copy())
	}
	return n
}

// IdentifierNode holds an identifier.
type IdentifierNode struct {
	NodeType
	Pos
	Ident string // The identifier's name.
}

// NewIdentifier returns a new IdentifierNode with the given identifier name.
func NewIdentifier(ident string) *IdentifierNode {
	return &IdentifierNode{NodeType: NodeIdentifier, Ident: ident}
}

// SetPos sets the position. NewIdentifier is a public method so we can't modify its signature.
// Chained for convenience.
// TODO: fix one day?
func (i *IdentifierNode) SetPos(pos Pos) *IdentifierNode {
	i.Pos = pos
	return i
}

func (i *IdentifierNode) String() string {
	return i.Ident
}

func (i *IdentifierNode) Copy() Node {
	return NewIdentifier(i.Ident).SetPos(i.Pos)
}

// VariableNode holds a list of variable names, possibly with chained field
// accesses. The dollar sign is part of the (first) name.
type VariableNode struct {
	NodeType
	Pos
	Ident []string // Variable name and fields in lexical order.
}

func newVariable(pos Pos, ident string) *VariableNode {
	return &VariableNode{NodeType: NodeVariable, Pos: pos, Ident: strings.Split(ident, ".")}
}

func (v *VariableNode) String() string {
	s := ""
	for i, id := range v.Ident {
		if i > 0 {
			s += "."
		}
		s += id
	}
	return s
}

func (v *VariableNode) Copy() Node {
	return &VariableNode{NodeType: NodeVariable, Pos: v.Pos, Ident: append([]string{}, v.Ident...)}
}

// DotNode holds the special identifier '.'.
type DotNode struct {
	Pos
}

func newDot(pos Pos) *DotNode {
	return &DotNode{Pos: pos}
}

func (d *DotNode) Type() NodeType {
	return NodeDot
}

func (d *DotNode) String() string {
	return "."
}

func (d *DotNode) Copy() Node {
	return newDot(d.Pos)
}

// NilNode holds the special identifier 'nil' representing an untyped nil constant.
type NilNode struct {
	Pos
}

func newNil(pos Pos) *NilNode {
	return &NilNode{Pos: pos}
}

func (n *NilNode) Type() NodeType {
	return NodeNil
}

func (n *NilNode) String() string {
	return "nil"
}

func (n *NilNode) Copy() Node {
	return newNil(n.Pos)
}

// FieldNode holds a field (identifier starting with '.').
// The names may be chained ('.x.y').
// The period is dropped from each ident.
type FieldNode struct {
	NodeType
	Pos
	Ident []string // The identifiers in lexical order.
}

func newField(pos Pos, ident string) *FieldNode {
	return &FieldNode{NodeType: NodeField, Pos: pos, Ident: strings.Split(ident[1:], ".")} // [1:] to drop leading period
}

func (f *FieldNode) String() string {
	s := ""
	for _, id := range f.Ident {
		s += "." + id
	}
	return s
}

func (f *FieldNode) Copy() Node {
	return &FieldNode{NodeType: NodeField, Pos: f.Pos, Ident: append([]string{}, f.Ident...)}
}

// ChainNode holds a term followed by a chain of field accesses (identifier starting with '.').
// The names may be chained ('.x.y').
// The periods are dropped from each ident.
type ChainNode struct {
	NodeType
	Pos
	Node  Node
	Field []string // The identifiers in lexical order.
}

func newChain(pos Pos, node Node) *ChainNode {
	return &ChainNode{NodeType: NodeChain, Pos: pos, Node: node}
}

// Add adds the named field (which should start with a period) to the end of the chain.
func (c *ChainNode) Add(field string) {
	if len(field) == 0 || field[0] != '.' {
		panic("no dot in field")
	}
	field = field[1:] // Remove leading dot.
	if field == "" {
		panic("empty field")
	}
	c.Field = append(c.Field, field)
}

func (c *ChainNode) String() string {
	s := c.Node.String()
	if _, ok := c.Node.(*PipeNode); ok {
		s = "(" + s + ")"
	}
	for _, field := range c.Field {
		s += "." + field
	}
	return s
}

func (c *ChainNode) Copy() Node {
	return &ChainNode{NodeType: NodeChain, Pos: c.Pos, Node: c.Node, Field: append([]string{}, c.Field...)}
}

// BoolNode holds a boolean constant.
type BoolNode struct {
	NodeType
	Pos
	True bool // The value of the boolean constant.
}

func newBool(pos Pos, true bool) *BoolNode {
	return &BoolNode{NodeType: NodeBool, Pos: pos, True: true}
}

func (b *BoolNode) String() string {
	if b.True {
		return "true"
	}
	return "false"
}

func (b *BoolNode) Copy() Node {
	return newBool(b.Pos, b.True)
}

// NumberNode holds a number: signed or unsigned integer, float, or complex.
// The value is parsed and stored under all the types that can represent the value.
// This simulates in a small amount of code the behavior of Go's ideal constants.
type NumberNode struct {
	NodeType
	Pos
	IsInt      bool       // Number has an integral value.
	IsUint     bool       // Number has an unsigned integral value.
	IsFloat    bool       // Number has a floating-point value.
	IsComplex  bool       // Number is complex.
	Int64      int64      // The signed integer value.
	Uint64     uint64     // The unsigned integer value.
	Float64    float64    // The floating-point value.
	Complex128 complex128 // The complex value.
	Text       string     // The original textual representation from the input.
}

func newNumber(pos Pos, text string, typ itemType) (*NumberNode, error) {
	n := &NumberNode{NodeType: NodeNumber, Pos: pos, Text: text}
	switch typ {
	case itemCharConstant:
		rune, _, tail, err := strconv.UnquoteChar(text[1:], text[0])
		if err != nil {
			return nil, err
		}
		if tail != "'" {
			return nil, fmt.Errorf("malformed character constant: %s", text)
		}
		n.Int64 = int64(rune)
		n.IsInt = true
		n.Uint64 = uint64(rune)
		n.IsUint = true
		n.Float64 = float64(rune) // odd but those are the rules.
		n.IsFloat = true
		return n, nil
	case itemComplex:
		// fmt.Sscan can parse the pair, so let it do the work.
		if _, err := fmt.Sscan(text, &n.Complex128); err != nil {
			return nil, err
		}
		n.IsComplex = true
		n.simplifyComplex()
		return n, nil
	}
	// Imaginary constants can only be complex unless they are zero.
	if len(text) > 0 && text[len(text)-1] == 'i' {
		f, err := strconv.ParseFloat(text[:len(text)-1], 64)
		if err == nil {
			n.IsComplex = true
			n.Complex128 = complex(0, f)
			n.simplifyComplex()
			return n, nil
		}
	}
	// Do integer test first so we get 0x123 etc.
	u, err := strconv.ParseUint(text, 0, 64) // will fail for -0; fixed below.
	if err == nil {
		n.IsUint = true
		n.Uint64 = u
	}
	i, err := strconv.ParseInt(text, 0, 64)
	if err == nil {
		n.IsInt = true
		n.Int64 = i
		if i == 0 {
			n.IsUint = true // in case of -0.
			n.Uint64 = u
		}
	}
	// If an integer extraction succeeded, promote the float.
	if n.IsInt {
		n.IsFloat = true
		n.Float64 = float64(n.Int64)
	} else if n.IsUint {
		n.IsFloat = true
		n.Float64 = float64(n.Uint64)
	} else {
		f, err := strconv.ParseFloat(text, 64)
		if err == nil {
			n.IsFloat = true
			n.Float64 = f
			// If a floating-point extraction succeeded, extract the int if needed.
			if !n.IsInt && float64(int64(f)) == f {
				n.IsInt = true
				n.Int64 = int64(f)
			}
			if !n.IsUint && float64(uint64(f)) == f {
				n.IsUint = true
				n.Uint64 = uint64(f)
			}
		}
	}
	if !n.IsInt && !n.IsUint && !n.IsFloat {
		return nil, fmt.Errorf("illegal number syntax: %q", text)
	}
	return n, nil
}

// simplifyComplex pulls out any other types that are represented by the complex number.
// These all require that the imaginary part be zero.
func (n *NumberNode) simplifyComplex() {
	n.IsFloat = imag(n.Complex128) == 0
	if n.IsFloat {
		n.Float64 = real(n.Complex128)
		n.IsInt = float64(int64(n.Float64)) == n.Float64
		if n.IsInt {
			n.Int64 = int64(n.Float64)
		}
		n.IsUint = float64(uint64(n.Float64)) == n.Float64
		if n.IsUint {
			n.Uint64 = uint64(n.Float64)
		}
	}
}

func (n *NumberNode) String() string {
	return n.Text
}

func (n *NumberNode) Copy() Node {
	nn := new(NumberNode)
	*nn = *n // Easy, fast, correct.
	return nn
}

// StringNode holds a string constant. The value has been "unquoted".
type StringNode struct {
	NodeType
	Pos
	Quoted string // The original text of the string, with quotes.
	Text   string // The string, after quote processing.
}

func newString(pos Pos, orig, text string) *StringNode {
	return &StringNode{NodeType: NodeString, Pos: pos, Quoted: orig, Text: text}
}

func (s *StringNode) String() string {
	return s.Quoted
}

func (s *StringNode) Copy() Node {
	return newString(s.Pos, s.Quoted, s.Text)
}

// endNode represents an {{end}} action.
// It does not appear in the final parse tree.
type endNode struct {
	Pos
}

func newEnd(pos Pos) *endNode {
	return &endNode{Pos: pos}
}

func (e *endNode) Type() NodeType {
	return nodeEnd
}

func (e *endNode) String() string {
	return "{{end}}"
}

func (e *endNode) Copy() Node {
	return newEnd(e.Pos)
}

// elseNode represents an {{else}} action. Does not appear in the final tree.
type elseNode struct {
	NodeType
	Pos
	Line int // The line number in the input (deprecated; kept for compatibility)
}

func newElse(pos Pos, line int) *elseNode {
	return &elseNode{NodeType: nodeElse, Pos: pos, Line: line}
}

func (e *elseNode) Type() NodeType {
	return nodeElse
}

func (e *elseNode) String() string {
	return "{{else}}"
}

func (e *elseNode) Copy() Node {
	return newElse(e.Pos, e.Line)
}

// BranchNode is the common representation of if, range, and with.
type BranchNode struct {
	NodeType
	Pos
	Line     int       // The line number in the input (deprecated; kept for compatibility)
	Pipe     *PipeNode // The pipeline to be evaluated.
	List     *ListNode // What to execute if the value is non-empty.
	ElseList *ListNode // What to execute if the value is empty (nil if absent).
}

func (b *BranchNode) String() string {
	name := ""
	switch b.NodeType {
	case NodeIf:
		name = "if"
	case NodeRange:
		name = "range"
	case NodeWith:
		name = "with"
	default:
		panic("unknown branch type")
	}
	if b.ElseList != nil {
		return fmt.Sprintf("{{%s %s}}%s{{else}}%s{{end}}", name, b.Pipe, b.List, b.ElseList)
	}
	return fmt.Sprintf("{{%s %s}}%s{{end}}", name, b.Pipe, b.List)
}

// IfNode represents an {{if}} action and its commands.
type IfNode struct {
	BranchNode
}

func newIf(pos Pos, line int, pipe *PipeNode, list, elseList *ListNode) *IfNode {
	return &IfNode{BranchNode{NodeType: NodeIf, Pos: pos, Line: line, Pipe: pipe, List: list, ElseList: elseList}}
}

func (i *IfNode) Copy() Node {
	return newIf(i.Pos, i.Line, i.Pipe.CopyPipe(), i.List.CopyList(), i.ElseList.CopyList())
}

// RangeNode represents a {{range}} action and its commands.
type RangeNode struct {
	BranchNode
}

func newRange(pos Pos, line int, pipe *PipeNode, list, elseList *ListNode) *RangeNode {
	return &RangeNode{BranchNode{NodeType: NodeRange, Pos: pos, Line: line, Pipe: pipe, List: list, ElseList: elseList}}
}

func (r *RangeNode) Copy() Node {
	return newRange(r.Pos, r.Line, r.Pipe.CopyPipe(), r.List.CopyList(), r.ElseList.CopyList())
}

// WithNode represents a {{with}} action and its commands.
type WithNode struct {
	BranchNode
}

func newWith(pos Pos, line int, pipe *PipeNode, list, elseList *ListNode) *WithNode {
	return &WithNode{BranchNode{NodeType: NodeWith, Pos: pos, Line: line, Pipe: pipe, List: list, ElseList: elseList}}
}

func (w *WithNode) Copy() Node {
	return newWith(w.Pos, w.Line, w.Pipe.CopyPipe(), w.List.CopyList(), w.ElseList.CopyList())
}

// TemplateNode represents a {{template}} action.
type TemplateNode struct {
	NodeType
	Pos
	Line int       // The line number in the input (deprecated; kept for compatibility)
	Name string    // The name of the template (unquoted).
	Pipe *PipeNode // The command to evaluate as dot for the template.
}

func newTemplate(pos Pos, line int, name string, pipe *PipeNode) *TemplateNode {
	return &TemplateNode{NodeType: NodeTemplate, Line: line, Pos: pos, Name: name, Pipe: pipe}
}

func (t *TemplateNode) String() string {
	if t.Pipe == nil {
		return fmt.Sprintf("{{template %q}}", t.Name)
	}
	return fmt.Sprintf("{{template %q %s}}", t.Name, t.Pipe)
}

func (t *TemplateNode) Copy() Node {
	return newTemplate(t.Pos, t.Line, t.Name, t.Pipe.CopyPipe())
}
