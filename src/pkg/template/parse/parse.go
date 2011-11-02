// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package parse builds parse trees for templates.  The grammar is defined
// in the documents for the template package.
package parse

import (
	"fmt"
	"runtime"
	"strconv"
	"unicode"
)

// Tree is the representation of a parsed template.
type Tree struct {
	Name string    // Name is the name of the template.
	Root *ListNode // Root is the top-level root of the parse tree.
	// Parsing only; cleared after parse.
	funcs     []map[string]interface{}
	lex       *lexer
	token     [2]item // two-token lookahead for parser.
	peekCount int
	vars      []string // variables defined at the moment.
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

// New allocates a new template with the given name.
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

// startParse starts the template parsing from the lexer.
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

// Parse parses the template definition string to construct an internal
// representation of the template for execution. If either action delimiter
// string is empty, the default ("{{" or "}}") is used.
func (t *Tree) Parse(s, leftDelim, rightDelim string, funcs ...map[string]interface{}) (tree *Tree, err error) {
	defer t.recover(&err)
	t.startParse(funcs, lex(t.Name, s, leftDelim, rightDelim))
	t.parse(true)
	t.stopParse()
	return t, nil
}

// parse is the helper for Parse.
// It triggers an error if we expect EOF but don't reach it.
func (t *Tree) parse(toEOF bool) (next Node) {
	t.Root, next = t.itemList(true)
	if toEOF && next != nil {
		t.errorf("unexpected %s", next)
	}
	return next
}

// itemList:
//	textOrAction*
// Terminates at EOF and at {{end}} or {{else}}, which is returned separately.
// The toEOF flag tells whether we expect to reach EOF.
func (t *Tree) itemList(toEOF bool) (list *ListNode, next Node) {
	list = newList()
	for t.peek().typ != itemEOF {
		n := t.textOrAction()
		switch n.Type() {
		case nodeEnd, nodeElse:
			return list, n
		}
		list.append(n)
	}
	if !toEOF {
		t.unexpected(t.next(), "input")
	}
	return list, nil
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
	list, next = t.itemList(false)
	switch next.Type() {
	case nodeEnd: //done
	case nodeElse:
		elseList, next = t.itemList(false)
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
