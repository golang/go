// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package parse builds parse trees for templates as defined by text/template
// and html/template. Clients should use those packages to construct templates
// rather than this one, which provides shared internal data structures not
// intended for general use.
package parse

import (
	"bytes"
	"fmt"
	"runtime"
	"strconv"
	"unicode"
)

// Tree is the representation of a single parsed template.
type Tree struct {
	Name string    // name of the template represented by the tree.
	Root *ListNode // top-level root of the tree.
	// Parsing only; cleared after parse.
	funcs     []map[string]interface{}
	lex       *lexer
	token     [2]item // two-token lookahead for parser.
	peekCount int
	vars      []string // variables defined at the moment.
}

// Parse returns a map from template name to parse.Tree, created by parsing the
// templates described in the argument string. The top-level template will be
// given the specified name. If an error is encountered, parsing stops and an
// empty map is returned with the error.
func Parse(name, text, leftDelim, rightDelim string, funcs ...map[string]interface{}) (treeSet map[string]*Tree, err error) {
	treeSet = make(map[string]*Tree)
	_, err = New(name).Parse(text, leftDelim, rightDelim, treeSet, funcs...)
	return
}

// next returns the next token.
func (t *Tree) next() item {
	if t.peekCount > 0 {
		t.peekCount--
	} else {
		t.token[0] = t.lex.nextItem()
	}
	return t.token[t.peekCount]
}

// backup backs the input stream up one token.
func (t *Tree) backup() {
	t.peekCount++
}

// backup2 backs the input stream up two tokens
func (t *Tree) backup2(t1 item) {
	t.token[1] = t1
	t.peekCount = 2
}

// peek returns but does not consume the next token.
func (t *Tree) peek() item {
	if t.peekCount > 0 {
		return t.token[t.peekCount-1]
	}
	t.peekCount = 1
	t.token[0] = t.lex.nextItem()
	return t.token[0]
}

// Parsing.

// New allocates a new parse tree with the given name.
func New(name string, funcs ...map[string]interface{}) *Tree {
	return &Tree{
		Name:  name,
		funcs: funcs,
	}
}

// errorf formats the error and terminates processing.
func (t *Tree) errorf(format string, args ...interface{}) {
	t.Root = nil
	format = fmt.Sprintf("template: %s:%d: %s", t.Name, t.lex.lineNumber(), format)
	panic(fmt.Errorf(format, args...))
}

// error terminates processing.
func (t *Tree) error(err error) {
	t.errorf("%s", err)
}

// expect consumes the next token and guarantees it has the required type.
func (t *Tree) expect(expected itemType, context string) item {
	token := t.next()
	if token.typ != expected {
		t.errorf("expected %s in %s; got %s", expected, context, token)
	}
	return token
}

// expectEither consumes the next token and guarantees it has one of the required types.
func (t *Tree) expectOneOf(expected1, expected2 itemType, context string) item {
	token := t.next()
	if token.typ != expected1 && token.typ != expected2 {
		t.errorf("expected %s or %s in %s; got %s", expected1, expected2, context, token)
	}
	return token
}

// unexpected complains about the token and terminates processing.
func (t *Tree) unexpected(token item, context string) {
	t.errorf("unexpected %s in %s", token, context)
}

// recover is the handler that turns panics into returns from the top level of Parse.
func (t *Tree) recover(errp *error) {
	e := recover()
	if e != nil {
		if _, ok := e.(runtime.Error); ok {
			panic(e)
		}
		if t != nil {
			t.stopParse()
		}
		*errp = e.(error)
	}
	return
}

// startParse initializes the parser, using the lexer.
func (t *Tree) startParse(funcs []map[string]interface{}, lex *lexer) {
	t.Root = nil
	t.lex = lex
	t.vars = []string{"$"}
	t.funcs = funcs
}

// stopParse terminates parsing.
func (t *Tree) stopParse() {
	t.lex = nil
	t.vars = nil
	t.funcs = nil
}

// atEOF returns true if, possibly after spaces, we're at EOF.
func (t *Tree) atEOF() bool {
	for {
		token := t.peek()
		switch token.typ {
		case itemEOF:
			return true
		case itemText:
			for _, r := range token.val {
				if !unicode.IsSpace(r) {
					return false
				}
			}
			t.next() // skip spaces.
			continue
		}
		break
	}
	return false
}

// Parse parses the template definition string to construct a representation of
// the template for execution. If either action delimiter string is empty, the
// default ("{{" or "}}") is used. Embedded template definitions are added to
// the treeSet map.
func (t *Tree) Parse(s, leftDelim, rightDelim string, treeSet map[string]*Tree, funcs ...map[string]interface{}) (tree *Tree, err error) {
	defer t.recover(&err)
	t.startParse(funcs, lex(t.Name, s, leftDelim, rightDelim))
	t.parse(treeSet)
	t.add(treeSet)
	t.stopParse()
	return t, nil
}

// add adds tree to the treeSet.
func (t *Tree) add(treeSet map[string]*Tree) {
	tree := treeSet[t.Name]
	if tree == nil || IsEmptyTree(tree.Root) {
		treeSet[t.Name] = t
		return
	}
	if !IsEmptyTree(t.Root) {
		t.errorf("template: multiple definition of template %q", t.Name)
	}
}

// IsEmptyTree reports whether this tree (node) is empty of everything but space.
func IsEmptyTree(n Node) bool {
	switch n := n.(type) {
	case *ActionNode:
	case *IfNode:
	case *ListNode:
		for _, node := range n.Nodes {
			if !IsEmptyTree(node) {
				return false
			}
		}
		return true
	case *RangeNode:
	case *TemplateNode:
	case *TextNode:
		return len(bytes.TrimSpace(n.Text)) == 0
	case *WithNode:
	default:
		panic("unknown node: " + n.String())
	}
	return false
}

// parse is the top-level parser for a template, essentially the same
// as itemList except it also parses {{define}} actions.
// It runs to EOF.
func (t *Tree) parse(treeSet map[string]*Tree) (next Node) {
	t.Root = newList()
	for t.peek().typ != itemEOF {
		if t.peek().typ == itemLeftDelim {
			delim := t.next()
			if t.next().typ == itemDefine {
				newT := New("definition") // name will be updated once we know it.
				newT.startParse(t.funcs, t.lex)
				newT.parseDefinition(treeSet)
				continue
			}
			t.backup2(delim)
		}
		n := t.textOrAction()
		if n.Type() == nodeEnd {
			t.errorf("unexpected %s", n)
		}
		t.Root.append(n)
	}
	return nil
}

// parseDefinition parses a {{define}} ...  {{end}} template definition and
// installs the definition in the treeSet map.  The "define" keyword has already
// been scanned.
func (t *Tree) parseDefinition(treeSet map[string]*Tree) {
	const context = "define clause"
	name := t.expectOneOf(itemString, itemRawString, context)
	var err error
	t.Name, err = strconv.Unquote(name.val)
	if err != nil {
		t.error(err)
	}
	t.expect(itemRightDelim, context)
	var end Node
	t.Root, end = t.itemList()
	if end.Type() != nodeEnd {
		t.errorf("unexpected %s in %s", end, context)
	}
	t.stopParse()
	t.add(treeSet)
}

// itemList:
//	textOrAction*
// Terminates at {{end}} or {{else}}, returned separately.
func (t *Tree) itemList() (list *ListNode, next Node) {
	list = newList()
	for t.peek().typ != itemEOF {
		n := t.textOrAction()
		switch n.Type() {
		case nodeEnd, nodeElse:
			return list, n
		}
		list.append(n)
	}
	t.errorf("unexpected EOF")
	return
}

// textOrAction:
//	text | action
func (t *Tree) textOrAction() Node {
	switch token := t.next(); token.typ {
	case itemText:
		return newText(token.val)
	case itemLeftDelim:
		return t.action()
	default:
		t.unexpected(token, "input")
	}
	return nil
}

// Action:
//	control
//	command ("|" command)*
// Left delim is past. Now get actions.
// First word could be a keyword such as range.
func (t *Tree) action() (n Node) {
	switch token := t.next(); token.typ {
	case itemElse:
		return t.elseControl()
	case itemEnd:
		return t.endControl()
	case itemIf:
		return t.ifControl()
	case itemRange:
		return t.rangeControl()
	case itemTemplate:
		return t.templateControl()
	case itemWith:
		return t.withControl()
	}
	t.backup()
	// Do not pop variables; they persist until "end".
	return newAction(t.lex.lineNumber(), t.pipeline("command"))
}

// Pipeline:
//	field or command
//	pipeline "|" pipeline
func (t *Tree) pipeline(context string) (pipe *PipeNode) {
	var decl []*VariableNode
	// Are there declarations?
	for {
		if v := t.peek(); v.typ == itemVariable {
			t.next()
			if next := t.peek(); next.typ == itemColonEquals || next.typ == itemChar {
				t.next()
				variable := newVariable(v.val)
				if len(variable.Ident) != 1 {
					t.errorf("illegal variable in declaration: %s", v.val)
				}
				decl = append(decl, variable)
				t.vars = append(t.vars, v.val)
				if next.typ == itemChar && next.val == "," {
					if context == "range" && len(decl) < 2 {
						continue
					}
					t.errorf("too many declarations in %s", context)
				}
			} else {
				t.backup2(v)
			}
		}
		break
	}
	pipe = newPipeline(t.lex.lineNumber(), decl)
	for {
		switch token := t.next(); token.typ {
		case itemRightDelim:
			if len(pipe.Cmds) == 0 {
				t.errorf("missing value for %s", context)
			}
			return
		case itemBool, itemCharConstant, itemComplex, itemDot, itemField, itemIdentifier,
			itemVariable, itemNumber, itemRawString, itemString:
			t.backup()
			pipe.append(t.command())
		default:
			t.unexpected(token, context)
		}
	}
	return
}

func (t *Tree) parseControl(context string) (lineNum int, pipe *PipeNode, list, elseList *ListNode) {
	lineNum = t.lex.lineNumber()
	defer t.popVars(len(t.vars))
	pipe = t.pipeline(context)
	var next Node
	list, next = t.itemList()
	switch next.Type() {
	case nodeEnd: //done
	case nodeElse:
		elseList, next = t.itemList()
		if next.Type() != nodeEnd {
			t.errorf("expected end; found %s", next)
		}
		elseList = elseList
	}
	return lineNum, pipe, list, elseList
}

// If:
//	{{if pipeline}} itemList {{end}}
//	{{if pipeline}} itemList {{else}} itemList {{end}}
// If keyword is past.
func (t *Tree) ifControl() Node {
	return newIf(t.parseControl("if"))
}

// Range:
//	{{range pipeline}} itemList {{end}}
//	{{range pipeline}} itemList {{else}} itemList {{end}}
// Range keyword is past.
func (t *Tree) rangeControl() Node {
	return newRange(t.parseControl("range"))
}

// With:
//	{{with pipeline}} itemList {{end}}
//	{{with pipeline}} itemList {{else}} itemList {{end}}
// If keyword is past.
func (t *Tree) withControl() Node {
	return newWith(t.parseControl("with"))
}

// End:
//	{{end}}
// End keyword is past.
func (t *Tree) endControl() Node {
	t.expect(itemRightDelim, "end")
	return newEnd()
}

// Else:
//	{{else}}
// Else keyword is past.
func (t *Tree) elseControl() Node {
	t.expect(itemRightDelim, "else")
	return newElse(t.lex.lineNumber())
}

// Template:
//	{{template stringValue pipeline}}
// Template keyword is past.  The name must be something that can evaluate
// to a string.
func (t *Tree) templateControl() Node {
	var name string
	switch token := t.next(); token.typ {
	case itemString, itemRawString:
		s, err := strconv.Unquote(token.val)
		if err != nil {
			t.error(err)
		}
		name = s
	default:
		t.unexpected(token, "template invocation")
	}
	var pipe *PipeNode
	if t.next().typ != itemRightDelim {
		t.backup()
		// Do not pop variables; they persist until "end".
		pipe = t.pipeline("template")
	}
	return newTemplate(t.lex.lineNumber(), name, pipe)
}

// command:
// space-separated arguments up to a pipeline character or right delimiter.
// we consume the pipe character but leave the right delim to terminate the action.
func (t *Tree) command() *CommandNode {
	cmd := newCommand()
Loop:
	for {
		switch token := t.next(); token.typ {
		case itemRightDelim:
			t.backup()
			break Loop
		case itemPipe:
			break Loop
		case itemError:
			t.errorf("%s", token.val)
		case itemIdentifier:
			if !t.hasFunction(token.val) {
				t.errorf("function %q not defined", token.val)
			}
			cmd.append(NewIdentifier(token.val))
		case itemDot:
			cmd.append(newDot())
		case itemVariable:
			cmd.append(t.useVar(token.val))
		case itemField:
			cmd.append(newField(token.val))
		case itemBool:
			cmd.append(newBool(token.val == "true"))
		case itemCharConstant, itemComplex, itemNumber:
			number, err := newNumber(token.val, token.typ)
			if err != nil {
				t.error(err)
			}
			cmd.append(number)
		case itemString, itemRawString:
			s, err := strconv.Unquote(token.val)
			if err != nil {
				t.error(err)
			}
			cmd.append(newString(token.val, s))
		default:
			t.unexpected(token, "command")
		}
	}
	if len(cmd.Args) == 0 {
		t.errorf("empty command")
	}
	return cmd
}

// hasFunction reports if a function name exists in the Tree's maps.
func (t *Tree) hasFunction(name string) bool {
	for _, funcMap := range t.funcs {
		if funcMap == nil {
			continue
		}
		if funcMap[name] != nil {
			return true
		}
	}
	return false
}

// popVars trims the variable list to the specified length
func (t *Tree) popVars(n int) {
	t.vars = t.vars[:n]
}

// useVar returns a node for a variable reference. It errors if the
// variable is not defined.
func (t *Tree) useVar(name string) Node {
	v := newVariable(name)
	for _, varName := range t.vars {
		if varName == v.Ident[0] {
			return v
		}
	}
	t.errorf("undefined variable %q", v.Ident[0])
	return nil
}
