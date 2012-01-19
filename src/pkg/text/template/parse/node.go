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

// A node is an element in the parse tree. The interface is trivial.
type Node interface {
	Type() NodeType
	String() string
}

// NodeType identifies the type of a parse tree node.
type NodeType int

// Type returns itself and provides an easy default implementation
// for embedding in a Node. Embedded in all non-trivial Nodes.
func (t NodeType) Type() NodeType {
	return t
}

const (
	NodeText       NodeType = iota // Plain text.
	NodeAction                     // A simple action such as field evaluation.
	NodeBool                       // A boolean constant.
	NodeCommand                    // An element of a pipeline.
	NodeDot                        // The cursor, dot.
	nodeElse                       // An else action. Not added to tree.
	nodeEnd                        // An end action. Not added to tree.
	NodeField                      // A field or method name.
	NodeIdentifier                 // An identifier; always a function name.
	NodeIf                         // An if action.
	NodeList                       // A list of Nodes.
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
	Nodes []Node // The element nodes in lexical order.
}

func newList() *ListNode {
	return &ListNode{NodeType: NodeList}
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

// TextNode holds plain text.
type TextNode struct {
	NodeType
	Text []byte // The text; may span newlines.
}

func newText(text string) *TextNode {
	return &TextNode{NodeType: NodeText, Text: []byte(text)}
}

func (t *TextNode) String() string {
	return fmt.Sprintf("%q", t.Text)
}

// PipeNode holds a pipeline with optional declaration
type PipeNode struct {
	NodeType
	Line int             // The line number in the input.
	Decl []*VariableNode // Variable declarations in lexical order.
	Cmds []*CommandNode  // The commands in lexical order.
}

func newPipeline(line int, decl []*VariableNode) *PipeNode {
	return &PipeNode{NodeType: NodePipe, Line: line, Decl: decl}
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

// ActionNode holds an action (something bounded by delimiters).
// Control actions have their own nodes; ActionNode represents simple
// ones such as field evaluations.
type ActionNode struct {
	NodeType
	Line int       // The line number in the input.
	Pipe *PipeNode // The pipeline in the action.
}

func newAction(line int, pipe *PipeNode) *ActionNode {
	return &ActionNode{NodeType: NodeAction, Line: line, Pipe: pipe}
}

func (a *ActionNode) String() string {
	return fmt.Sprintf("{{%s}}", a.Pipe)

}

// CommandNode holds a command (a pipeline inside an evaluating action).
type CommandNode struct {
	NodeType
	Args []Node // Arguments in lexical order: Identifier, field, or constant.
}

func newCommand() *CommandNode {
	return &CommandNode{NodeType: NodeCommand}
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
		s += arg.String()
	}
	return s
}

// IdentifierNode holds an identifier.
type IdentifierNode struct {
	NodeType
	Ident string // The identifier's name.
}

// NewIdentifier returns a new IdentifierNode with the given identifier name.
func NewIdentifier(ident string) *IdentifierNode {
	return &IdentifierNode{NodeType: NodeIdentifier, Ident: ident}
}

func (i *IdentifierNode) String() string {
	return i.Ident
}

// VariableNode holds a list of variable names. The dollar sign is
// part of the name.
type VariableNode struct {
	NodeType
	Ident []string // Variable names in lexical order.
}

func newVariable(ident string) *VariableNode {
	return &VariableNode{NodeType: NodeVariable, Ident: strings.Split(ident, ".")}
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

// DotNode holds the special identifier '.'. It is represented by a nil pointer.
type DotNode bool

func newDot() *DotNode {
	return nil
}

func (d *DotNode) Type() NodeType {
	return NodeDot
}

func (d *DotNode) String() string {
	return "."
}

// FieldNode holds a field (identifier starting with '.').
// The names may be chained ('.x.y').
// The period is dropped from each ident.
type FieldNode struct {
	NodeType
	Ident []string // The identifiers in lexical order.
}

func newField(ident string) *FieldNode {
	return &FieldNode{NodeType: NodeField, Ident: strings.Split(ident[1:], ".")} // [1:] to drop leading period
}

func (f *FieldNode) String() string {
	s := ""
	for _, id := range f.Ident {
		s += "." + id
	}
	return s
}

// BoolNode holds a boolean constant.
type BoolNode struct {
	NodeType
	True bool // The value of the boolean constant.
}

func newBool(true bool) *BoolNode {
	return &BoolNode{NodeType: NodeBool, True: true}
}

func (b *BoolNode) String() string {
	if b.True {
		return "true"
	}
	return "false"
}

// NumberNode holds a number: signed or unsigned integer, float, or complex.
// The value is parsed and stored under all the types that can represent the value.
// This simulates in a small amount of code the behavior of Go's ideal constants.
type NumberNode struct {
	NodeType
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

func newNumber(text string, typ itemType) (*NumberNode, error) {
	n := &NumberNode{NodeType: NodeNumber, Text: text}
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

// StringNode holds a string constant. The value has been "unquoted".
type StringNode struct {
	NodeType
	Quoted string // The original text of the string, with quotes.
	Text   string // The string, after quote processing.
}

func newString(orig, text string) *StringNode {
	return &StringNode{NodeType: NodeString, Quoted: orig, Text: text}
}

func (s *StringNode) String() string {
	return s.Quoted
}

// endNode represents an {{end}} action. It is represented by a nil pointer.
// It does not appear in the final parse tree.
type endNode bool

func newEnd() *endNode {
	return nil
}

func (e *endNode) Type() NodeType {
	return nodeEnd
}

func (e *endNode) String() string {
	return "{{end}}"
}

// elseNode represents an {{else}} action. Does not appear in the final tree.
type elseNode struct {
	NodeType
	Line int // The line number in the input.
}

func newElse(line int) *elseNode {
	return &elseNode{NodeType: nodeElse, Line: line}
}

func (e *elseNode) Type() NodeType {
	return nodeElse
}

func (e *elseNode) String() string {
	return "{{else}}"
}

// BranchNode is the common representation of if, range, and with.
type BranchNode struct {
	NodeType
	Line     int       // The line number in the input.
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

func newIf(line int, pipe *PipeNode, list, elseList *ListNode) *IfNode {
	return &IfNode{BranchNode{NodeType: NodeIf, Line: line, Pipe: pipe, List: list, ElseList: elseList}}
}

// RangeNode represents a {{range}} action and its commands.
type RangeNode struct {
	BranchNode
}

func newRange(line int, pipe *PipeNode, list, elseList *ListNode) *RangeNode {
	return &RangeNode{BranchNode{NodeType: NodeRange, Line: line, Pipe: pipe, List: list, ElseList: elseList}}
}

// WithNode represents a {{with}} action and its commands.
type WithNode struct {
	BranchNode
}

func newWith(line int, pipe *PipeNode, list, elseList *ListNode) *WithNode {
	return &WithNode{BranchNode{NodeType: NodeWith, Line: line, Pipe: pipe, List: list, ElseList: elseList}}
}

// TemplateNode represents a {{template}} action.
type TemplateNode struct {
	NodeType
	Line int       // The line number in the input.
	Name string    // The name of the template (unquoted).
	Pipe *PipeNode // The command to evaluate as dot for the template.
}

func newTemplate(line int, name string, pipe *PipeNode) *TemplateNode {
	return &TemplateNode{NodeType: NodeTemplate, Line: line, Name: name, Pipe: pipe}
}

func (t *TemplateNode) String() string {
	if t.Pipe == nil {
		return fmt.Sprintf("{{template %q}}", t.Name)
	}
	return fmt.Sprintf("{{template %q %s}}", t.Name, t.Pipe)
}
