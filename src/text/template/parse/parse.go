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
	"strings"
)

// Tree is the representation of a single parsed template.
type Tree struct {
	Name      string    // name of the template represented by the tree.
	ParseName string    // name of the top-level template during parsing, for error messages.
	Root      *ListNode // top-level root of the tree.
	Mode      Mode      // parsing mode.
	text      string    // text parsed to create the template (or its parent)
	// Parsing only; cleared after parse.
	funcs      []map[string]any
	lex        *lexer
	token      [3]item // three-token lookahead for parser.
	peekCount  int
	vars       []string // variables defined at the moment.
	treeSet    map[string]*Tree
	actionLine int // line of left delim starting action
	rangeDepth int
	stackDepth int // depth of nested parenthesized expressions
}

// A mode value is a set of flags (or 0). Modes control parser behavior.
type Mode uint

const (
	ParseComments Mode = 1 << iota // parse comments and add them to AST
	SkipFuncCheck                  // do not check that functions are defined
)

// maxStackDepth is the maximum depth permitted for nested
// parenthesized expressions.
var maxStackDepth = 10000

// maxPipelineCmds is the maximum number of commands permitted
// for a single template's pipeline.
var maxPipelineCmds = 100000

// init reduces maxStackDepth for WebAssembly due to its smaller stack size.
func init() {
	if runtime.GOARCH == "wasm" {
		maxStackDepth = 1000
	}
}

// Copy returns a copy of the [Tree]. Any parsing state is discarded.
func (t *Tree) Copy() *Tree {
	if t == nil {
		return nil
	}
	return &Tree{
		Name:      t.Name,
		ParseName: t.ParseName,
		Root:      t.Root.CopyList(),
		text:      t.text,
	}
}

// Parse returns a map from template name to [Tree], created by parsing the
// templates described in the argument string. The top-level template will be
// given the specified name. If an error is encountered, parsing stops and an
// empty map is returned with the error.
func Parse(name, text, leftDelim, rightDelim string, funcs ...map[string]any) (map[string]*Tree, error) {
	treeSet := make(map[string]*Tree)
	t := New(name)
	t.text = text
	_, err := t.Parse(text, leftDelim, rightDelim, treeSet, funcs...)
	return treeSet, err
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

// backup2 backs the input stream up two tokens.
// The zeroth token is already there.
func (t *Tree) backup2(t1 item) {
	t.token[1] = t1
	t.peekCount = 2
}

// backup3 backs the input stream up three tokens
// The zeroth token is already there.
func (t *Tree) backup3(t2, t1 item) { // Reverse order: we're pushing back.
	t.token[1] = t1
	t.token[2] = t2
	t.peekCount = 3
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

// nextNonSpace returns the next non-space token.
func (t *Tree) nextNonSpace() (token item) {
	for {
		token = t.next()
		if token.typ != itemSpace {
			break
		}
	}
	return token
}

// peekNonSpace returns but does not consume the next non-space token.
func (t *Tree) peekNonSpace() item {
	token := t.nextNonSpace()
	t.backup()
	return token
}

// Parsing.

// New allocates a new parse tree with the given name.
func New(name string, funcs ...map[string]any) *Tree {
	return &Tree{
		Name:  name,
		funcs: funcs,
	}
}

// ErrorContext returns a textual representation of the location of the node in the input text.
// The receiver is only used when the node does not have a pointer to the tree inside,
// which can occur in old code.
func (t *Tree) ErrorContext(n Node) (location, context string) {
	pos := int(n.Position())
	tree := n.tree()
	if tree == nil {
		tree = t
	}
	text := tree.text[:pos]
	byteNum := strings.LastIndex(text, "\n")
	if byteNum == -1 {
		byteNum = pos // On first line.
	} else {
		byteNum++ // After the newline.
		byteNum = pos - byteNum
	}
	lineNum := 1 + strings.Count(text, "\n")
	context = n.String()
	return fmt.Sprintf("%s:%d:%d", tree.ParseName, lineNum, byteNum), context
}

// errorf formats the error and terminates processing.
func (t *Tree) errorf(format string, args ...any) {
	t.Root = nil
	format = fmt.Sprintf("template: %s:%d: %s", t.ParseName, t.token[0].line, format)
	panic(fmt.Errorf(format, args...))
}

// error terminates processing.
func (t *Tree) error(err error) {
	t.errorf("%s", err)
}

// expect consumes the next token and guarantees it has the required type.
func (t *Tree) expect(expected itemType, context string) item {
	token := t.nextNonSpace()
	if token.typ != expected {
		t.unexpected(token, context)
	}
	return token
}

// expectOneOf consumes the next token and guarantees it has one of the required types.
func (t *Tree) expectOneOf(expected1, expected2 itemType, context string) item {
	token := t.nextNonSpace()
	if token.typ != expected1 && token.typ != expected2 {
		t.unexpected(token, context)
	}
	return token
}

// unexpected complains about the token and terminates processing.
func (t *Tree) unexpected(token item, context string) {
	if token.typ == itemError {
		extra := ""
		if t.actionLine != 0 && t.actionLine != token.line {
			extra = fmt.Sprintf(" in action started at %s:%d", t.ParseName, t.actionLine)
			if strings.HasSuffix(token.val, " action") {
				extra = extra[len(" in action"):] // avoid "action in action"
			}
		}
		t.errorf("%s%s", token, extra)
	}
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
}

// startParse initializes the parser, using the lexer.
func (t *Tree) startParse(funcs []map[string]any, lex *lexer, treeSet map[string]*Tree) {
	t.Root = nil
	t.lex = lex
	t.vars = []string{"$"}
	t.funcs = funcs
	t.treeSet = treeSet
	t.stackDepth = 0
	lex.options = lexOptions{
		emitComment: t.Mode&ParseComments != 0,
		breakOK:     !t.hasFunction("break"),
		continueOK:  !t.hasFunction("continue"),
	}
}

// stopParse terminates parsing.
func (t *Tree) stopParse() {
	t.lex = nil
	t.vars = nil
	t.funcs = nil
	t.treeSet = nil
}

// Parse parses the template definition string to construct a representation of
// the template for execution. If either action delimiter string is empty, the
// default ("{{" or "}}") is used. Embedded template definitions are added to
// the treeSet map.
func (t *Tree) Parse(text, leftDelim, rightDelim string, treeSet map[string]*Tree, funcs ...map[string]any) (tree *Tree, err error) {
	defer t.recover(&err)
	t.ParseName = t.Name
	lexer := lex(t.Name, text, leftDelim, rightDelim)
	t.startParse(funcs, lexer, treeSet)
	t.text = text
	t.parse()
	t.add()
	t.stopParse()
	return t, nil
}

// add adds tree to t.treeSet.
func (t *Tree) add() {
	tree := t.treeSet[t.Name]
	if tree == nil || IsEmptyTree(tree.Root) {
		t.treeSet[t.Name] = t
		return
	}
	if !IsEmptyTree(t.Root) {
		t.errorf("template: multiple definition of template %q", t.Name)
	}
}

// IsEmptyTree reports whether this tree (node) is empty of everything but space or comments.
func IsEmptyTree(n Node) bool {
	switch n := n.(type) {
	case nil:
		return true
	case *ActionNode:
	case *CommentNode:
		return true
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
func (t *Tree) parse() {
	t.Root = t.newList(t.peek().pos)
	for t.peek().typ != itemEOF {
		if t.peek().typ == itemLeftDelim {
			delim := t.next()
			if t.nextNonSpace().typ == itemDefine {
				newT := New("definition") // name will be updated once we know it.
				newT.text = t.text
				newT.Mode = t.Mode
				newT.ParseName = t.ParseName
				newT.startParse(t.funcs, t.lex, t.treeSet)
				newT.parseDefinition()
				continue
			}
			t.backup2(delim)
		}
		switch n := t.textOrAction(); n.Type() {
		case nodeEnd, nodeElse:
			t.errorf("unexpected %s", n)
		default:
			t.Root.append(n)
		}
	}
}

// parseDefinition parses a {{define}} ...  {{end}} template definition and
// installs the definition in t.treeSet. The "define" keyword has already
// been scanned.
func (t *Tree) parseDefinition() {
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
	t.add()
	t.stopParse()
}

// itemList:
//
//	textOrAction*
//
// Terminates at {{end}} or {{else}}, returned separately.
func (t *Tree) itemList() (list *ListNode, next Node) {
	list = t.newList(t.peekNonSpace().pos)
	for t.peekNonSpace().typ != itemEOF {
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
//
//	text | comment | action
func (t *Tree) textOrAction() Node {
	switch token := t.nextNonSpace(); token.typ {
	case itemText:
		return t.newText(token.pos, token.val)
	case itemLeftDelim:
		t.actionLine = token.line
		defer t.clearActionLine()
		return t.action()
	case itemComment:
		return t.newComment(token.pos, token.val)
	default:
		t.unexpected(token, "input")
	}
	return nil
}

func (t *Tree) clearActionLine() {
	t.actionLine = 0
}

// Action:
//
//	control
//	command ("|" command)*
//
// Left delim is past. Now get actions.
// First word could be a keyword such as range.
func (t *Tree) action() (n Node) {
	switch token := t.nextNonSpace(); token.typ {
	case itemBlock:
		return t.blockControl()
	case itemBreak:
		return t.breakControl(token.pos, token.line)
	case itemContinue:
		return t.continueControl(token.pos, token.line)
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
	token := t.peek()
	// Do not pop variables; they persist until "end".
	return t.newAction(token.pos, token.line, t.pipeline("command", itemRightDelim))
}

// Break:
//
//	{{break}}
//
// Break keyword is past.
func (t *Tree) breakControl(pos Pos, line int) Node {
	if token := t.nextNonSpace(); token.typ != itemRightDelim {
		t.unexpected(token, "{{break}}")
	}
	if t.rangeDepth == 0 {
		t.errorf("{{break}} outside {{range}}")
	}
	return t.newBreak(pos, line)
}

// Continue:
//
//	{{continue}}
//
// Continue keyword is past.
func (t *Tree) continueControl(pos Pos, line int) Node {
	if token := t.nextNonSpace(); token.typ != itemRightDelim {
		t.unexpected(token, "{{continue}}")
	}
	if t.rangeDepth == 0 {
		t.errorf("{{continue}} outside {{range}}")
	}
	return t.newContinue(pos, line)
}

// Pipeline:
//
//	declarations? command ('|' command)*
func (t *Tree) pipeline(context string, end itemType) (pipe *PipeNode) {
	token := t.peekNonSpace()
	pipe = t.newPipeline(token.pos, token.line, nil)
	// Are there declarations or assignments?
decls:
	if v := t.peekNonSpace(); v.typ == itemVariable {
		t.next()
		// Since space is a token, we need 3-token look-ahead here in the worst case:
		// in "$x foo" we need to read "foo" (as opposed to ":=") to know that $x is an
		// argument variable rather than a declaration. So remember the token
		// adjacent to the variable so we can push it back if necessary.
		tokenAfterVariable := t.peek()
		next := t.peekNonSpace()
		switch {
		case next.typ == itemAssign, next.typ == itemDeclare:
			pipe.IsAssign = next.typ == itemAssign
			t.nextNonSpace()
			pipe.Decl = append(pipe.Decl, t.newVariable(v.pos, v.val))
			t.vars = append(t.vars, v.val)
		case next.typ == itemChar && next.val == ",":
			t.nextNonSpace()
			pipe.Decl = append(pipe.Decl, t.newVariable(v.pos, v.val))
			t.vars = append(t.vars, v.val)
			if context == "range" && len(pipe.Decl) < 2 {
				switch t.peekNonSpace().typ {
				case itemVariable, itemRightDelim, itemRightParen:
					// second initialized variable in a range pipeline
					goto decls
				default:
					t.errorf("range can only initialize variables")
				}
			}
			t.errorf("too many declarations in %s", context)
		case tokenAfterVariable.typ == itemSpace:
			t.backup3(v, tokenAfterVariable)
		default:
			t.backup2(v)
		}
	}
	for {
		switch token := t.nextNonSpace(); token.typ {
		case end:
			// At this point, the pipeline is complete
			t.checkPipeline(pipe, context)
			return
		case itemBool, itemCharConstant, itemComplex, itemDot, itemField, itemIdentifier,
			itemNumber, itemNil, itemRawString, itemString, itemVariable, itemLeftParen:
			t.backup()
			pipe.append(t.command())
			if len(pipe.Cmds) > maxPipelineCmds {
				t.errorf("pipeline too long")
			}
		default:
			t.unexpected(token, context)
		}
	}
}

func (t *Tree) checkPipeline(pipe *PipeNode, context string) {
	// Reject empty pipelines
	if len(pipe.Cmds) == 0 {
		t.errorf("missing value for %s", context)
	}
	// Only the first command of a pipeline can start with a non executable operand
	for i, c := range pipe.Cmds[1:] {
		switch c.Args[0].Type() {
		case NodeBool, NodeDot, NodeNil, NodeNumber, NodeString:
			// With A|B|C, pipeline stage 2 is B
			t.errorf("non executable command in pipeline stage %d", i+2)
		}
	}
}

func (t *Tree) parseControl(context string) (pos Pos, line int, pipe *PipeNode, list, elseList *ListNode) {
	defer t.popVars(len(t.vars))
	pipe = t.pipeline(context, itemRightDelim)
	if context == "range" {
		t.rangeDepth++
	}
	var next Node
	list, next = t.itemList()
	if context == "range" {
		t.rangeDepth--
	}
	switch next.Type() {
	case nodeEnd: //done
	case nodeElse:
		// Special case for "else if" and "else with".
		// If the "else" is followed immediately by an "if" or "with",
		// the elseControl will have left the "if" or "with" token pending. Treat
		//	{{if a}}_{{else if b}}_{{end}}
		//  {{with a}}_{{else with b}}_{{end}}
		// as
		//	{{if a}}_{{else}}{{if b}}_{{end}}{{end}}
		//  {{with a}}_{{else}}{{with b}}_{{end}}{{end}}.
		// To do this, parse the "if" or "with" as usual and stop at it {{end}};
		// the subsequent{{end}} is assumed. This technique works even for long if-else-if chains.
		if context == "if" && t.peek().typ == itemIf {
			t.next() // Consume the "if" token.
			elseList = t.newList(next.Position())
			elseList.append(t.ifControl())
		} else if context == "with" && t.peek().typ == itemWith {
			t.next()
			elseList = t.newList(next.Position())
			elseList.append(t.withControl())
		} else {
			elseList, next = t.itemList()
			if next.Type() != nodeEnd {
				t.errorf("expected end; found %s", next)
			}
		}
	}
	return pipe.Position(), pipe.Line, pipe, list, elseList
}

// If:
//
//	{{if pipeline}} itemList {{end}}
//	{{if pipeline}} itemList {{else}} itemList {{end}}
//
// If keyword is past.
func (t *Tree) ifControl() Node {
	return t.newIf(t.parseControl("if"))
}

// Range:
//
//	{{range pipeline}} itemList {{end}}
//	{{range pipeline}} itemList {{else}} itemList {{end}}
//
// Range keyword is past.
func (t *Tree) rangeControl() Node {
	r := t.newRange(t.parseControl("range"))
	return r
}

// With:
//
//	{{with pipeline}} itemList {{end}}
//	{{with pipeline}} itemList {{else}} itemList {{end}}
//
// If keyword is past.
func (t *Tree) withControl() Node {
	return t.newWith(t.parseControl("with"))
}

// End:
//
//	{{end}}
//
// End keyword is past.
func (t *Tree) endControl() Node {
	return t.newEnd(t.expect(itemRightDelim, "end").pos)
}

// Else:
//
//	{{else}}
//
// Else keyword is past.
func (t *Tree) elseControl() Node {
	peek := t.peekNonSpace()
	// The "{{else if ... " and "{{else with ..." will be
	// treated as "{{else}}{{if ..." and "{{else}}{{with ...".
	// So return the else node here.
	if peek.typ == itemIf || peek.typ == itemWith {
		return t.newElse(peek.pos, peek.line)
	}
	token := t.expect(itemRightDelim, "else")
	return t.newElse(token.pos, token.line)
}

// Block:
//
//	{{block stringValue pipeline}}
//
// Block keyword is past.
// The name must be something that can evaluate to a string.
// The pipeline is mandatory.
func (t *Tree) blockControl() Node {
	const context = "block clause"

	token := t.nextNonSpace()
	name := t.parseTemplateName(token, context)
	pipe := t.pipeline(context, itemRightDelim)

	block := New(name) // name will be updated once we know it.
	block.text = t.text
	block.Mode = t.Mode
	block.ParseName = t.ParseName
	block.startParse(t.funcs, t.lex, t.treeSet)
	var end Node
	block.Root, end = block.itemList()
	if end.Type() != nodeEnd {
		t.errorf("unexpected %s in %s", end, context)
	}
	block.add()
	block.stopParse()

	return t.newTemplate(token.pos, token.line, name, pipe)
}

// Template:
//
//	{{template stringValue pipeline}}
//
// Template keyword is past. The name must be something that can evaluate
// to a string.
func (t *Tree) templateControl() Node {
	const context = "template clause"
	token := t.nextNonSpace()
	name := t.parseTemplateName(token, context)
	var pipe *PipeNode
	if t.nextNonSpace().typ != itemRightDelim {
		t.backup()
		// Do not pop variables; they persist until "end".
		pipe = t.pipeline(context, itemRightDelim)
	}
	return t.newTemplate(token.pos, token.line, name, pipe)
}

func (t *Tree) parseTemplateName(token item, context string) (name string) {
	switch token.typ {
	case itemString, itemRawString:
		s, err := strconv.Unquote(token.val)
		if err != nil {
			t.error(err)
		}
		name = s
	default:
		t.unexpected(token, context)
	}
	return
}

// command:
//
//	operand (space operand)*
//
// space-separated arguments up to a pipeline character or right delimiter.
// we consume the pipe character but leave the right delim to terminate the action.
func (t *Tree) command() *CommandNode {
	cmd := t.newCommand(t.peekNonSpace().pos)
	for {
		t.peekNonSpace() // skip leading spaces.
		operand := t.operand()
		if operand != nil {
			cmd.append(operand)
		}
		switch token := t.next(); token.typ {
		case itemSpace:
			continue
		case itemRightDelim, itemRightParen:
			t.backup()
		case itemPipe:
			// nothing here; break loop below
		default:
			t.unexpected(token, "operand")
		}
		break
	}
	if len(cmd.Args) == 0 {
		t.errorf("empty command")
	}
	return cmd
}

// operand:
//
//	term .Field*
//
// An operand is a space-separated component of a command,
// a term possibly followed by field accesses.
// A nil return means the next item is not an operand.
func (t *Tree) operand() Node {
	node := t.term()
	if node == nil {
		return nil
	}
	if t.peek().typ == itemField {
		chain := t.newChain(t.peek().pos, node)
		for t.peek().typ == itemField {
			chain.Add(t.next().val)
		}
		// Compatibility with original API: If the term is of type NodeField
		// or NodeVariable, just put more fields on the original.
		// Otherwise, keep the Chain node.
		// Obvious parsing errors involving literal values are detected here.
		// More complex error cases will have to be handled at execution time.
		switch node.Type() {
		case NodeField:
			node = t.newField(chain.Position(), chain.String())
		case NodeVariable:
			node = t.newVariable(chain.Position(), chain.String())
		case NodeBool, NodeString, NodeNumber, NodeNil, NodeDot:
			t.errorf("unexpected . after term %q", node.String())
		default:
			node = chain
		}
	}
	return node
}

// term:
//
//	literal (number, string, nil, boolean)
//	function (identifier)
//	.
//	.Field
//	$
//	'(' pipeline ')'
//
// A term is a simple "expression".
// A nil return means the next item is not a term.
func (t *Tree) term() Node {
	switch token := t.nextNonSpace(); token.typ {
	case itemIdentifier:
		checkFunc := t.Mode&SkipFuncCheck == 0
		if checkFunc && !t.hasFunction(token.val) {
			t.errorf("function %q not defined", token.val)
		}
		return NewIdentifier(token.val).SetTree(t).SetPos(token.pos)
	case itemDot:
		return t.newDot(token.pos)
	case itemNil:
		return t.newNil(token.pos)
	case itemVariable:
		return t.useVar(token.pos, token.val)
	case itemField:
		return t.newField(token.pos, token.val)
	case itemBool:
		return t.newBool(token.pos, token.val == "true")
	case itemCharConstant, itemComplex, itemNumber:
		number, err := t.newNumber(token.pos, token.val, token.typ)
		if err != nil {
			t.error(err)
		}
		return number
	case itemLeftParen:
		if t.stackDepth >= maxStackDepth {
			t.errorf("max expression depth exceeded")
		}
		t.stackDepth++
		defer func() { t.stackDepth-- }()
		return t.pipeline("parenthesized pipeline", itemRightParen)
	case itemString, itemRawString:
		s, err := strconv.Unquote(token.val)
		if err != nil {
			t.error(err)
		}
		return t.newString(token.pos, token.val, s)
	}
	t.backup()
	return nil
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
func (t *Tree) useVar(pos Pos, name string) Node {
	v := t.newVariable(pos, name)
	for _, varName := range t.vars {
		if varName == v.Ident[0] {
			return v
		}
	}
	t.errorf("undefined variable %q", v.Ident[0])
	return nil
}
