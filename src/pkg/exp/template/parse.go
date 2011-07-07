// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes"
	"fmt"
	"os"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"unicode"
)

// Template is the representation of a parsed template.
type Template struct {
	name  string
	root  *listNode
	funcs map[string]reflect.Value
	// Parsing only; cleared after parse.
	set      *Set
	lex      *lexer
	token    item // token lookahead for parser
	havePeek bool
}

// next returns the next token.
func (t *Template) next() item {
	if t.havePeek {
		t.havePeek = false
	} else {
		t.token = t.lex.nextItem()
	}
	return t.token
}

// backup backs the input stream up one token.
func (t *Template) backup() {
	t.havePeek = true
}

// peek returns but does not consume the next token.
func (t *Template) peek() item {
	if t.havePeek {
		return t.token
	}
	t.token = t.lex.nextItem()
	t.havePeek = true
	return t.token
}

// A node is an element in the parse tree. The interface is trivial.
type node interface {
	typ() nodeType
	String() string
}

type nodeType int

func (t nodeType) typ() nodeType {
	return t
}

const (
	nodeText nodeType = iota
	nodeAction
	nodeCommand
	nodeDot
	nodeElse
	nodeEnd
	nodeField
	nodeIdentifier
	nodeIf
	nodeList
	nodeNumber
	nodeRange
	nodeString
	nodeTemplate
	nodeWith
)

// Nodes.

// listNode holds a sequence of nodes.
type listNode struct {
	nodeType
	nodes []node
}

func newList() *listNode {
	return &listNode{nodeType: nodeList}
}

func (l *listNode) append(n node) {
	l.nodes = append(l.nodes, n)
}

func (l *listNode) String() string {
	b := new(bytes.Buffer)
	fmt.Fprint(b, "[")
	for _, n := range l.nodes {
		fmt.Fprint(b, n)
	}
	fmt.Fprint(b, "]")
	return b.String()
}

// textNode holds plain text.
type textNode struct {
	nodeType
	text []byte
}

func newText(text string) *textNode {
	return &textNode{nodeType: nodeText, text: []byte(text)}
}

func (t *textNode) String() string {
	return fmt.Sprintf("(text: %q)", t.text)
}

// actionNode holds an action (something bounded by delimiters).
type actionNode struct {
	nodeType
	line     int
	pipeline []*commandNode
}

func newAction(line int, pipeline []*commandNode) *actionNode {
	return &actionNode{nodeType: nodeAction, line: line, pipeline: pipeline}
}

func (a *actionNode) append(command *commandNode) {
	a.pipeline = append(a.pipeline, command)
}

func (a *actionNode) String() string {
	return fmt.Sprintf("(action: %v)", a.pipeline)
}

// commandNode holds a command (a pipeline inside an evaluating action).
type commandNode struct {
	nodeType
	args []node // identifier, string, or number
}

func newCommand() *commandNode {
	return &commandNode{nodeType: nodeCommand}
}

func (c *commandNode) append(arg node) {
	c.args = append(c.args, arg)
}

func (c *commandNode) String() string {
	return fmt.Sprintf("(command: %v)", c.args)
}

// identifierNode holds an identifier.
type identifierNode struct {
	nodeType
	ident string
}

func newIdentifier(ident string) *identifierNode {
	return &identifierNode{nodeType: nodeIdentifier, ident: ident}
}

func (i *identifierNode) String() string {
	return fmt.Sprintf("I=%s", i.ident)
}

// dotNode holds the special identifier '.'. It is represented by a nil pointer.
type dotNode bool

func newDot() *dotNode {
	return nil
}

func (d *dotNode) typ() nodeType {
	return nodeDot
}

func (d *dotNode) String() string {
	return "{{<.>}}"
}

// fieldNode holds a field (identifier starting with '.').
// The names may be chained ('.x.y').
// The period is dropped from each ident.
type fieldNode struct {
	nodeType
	ident []string
}

func newField(ident string) *fieldNode {
	return &fieldNode{nodeType: nodeField, ident: strings.Split(ident[1:], ".")} // [1:] to drop leading period
}

func (f *fieldNode) String() string {
	return fmt.Sprintf("F=%s", f.ident)
}

// boolNode holds a boolean constant.
type boolNode struct {
	nodeType
	true bool
}

func newBool(true bool) *boolNode {
	return &boolNode{nodeType: nodeString, true: true}
}

func (b *boolNode) String() string {
	if b.true {
		return fmt.Sprintf("B=true")
	}
	return fmt.Sprintf("B=false")
}

// numberNode holds a number, signed or unsigned integer, floating, or complex.
// The value is parsed and stored under all the types that can represent the value.
// This simulates in a small amount of code the behavior of Go's ideal constants.
type numberNode struct {
	nodeType
	isInt      bool // number has an integral value
	isUint     bool // number has an unsigned integral value
	isFloat    bool // number has a floating-point value
	isComplex  bool // number is complex
	int64           // the signed integer value
	uint64          // the unsigned integer value
	float64         // the floating-point value
	complex128      // the complex value
	text       string
}

func newNumber(text string, isComplex bool) (*numberNode, os.Error) {
	n := &numberNode{nodeType: nodeNumber, text: text}
	if isComplex {
		// fmt.Sscan can parse the pair, so let it do the work.
		if _, err := fmt.Sscan(text, &n.complex128); err != nil {
			return nil, err
		}
		n.isComplex = true
		n.simplifyComplex()
		return n, nil
	}
	// Imaginary constants can only be complex unless they are zero.
	if len(text) > 0 && text[len(text)-1] == 'i' {
		f, err := strconv.Atof64(text[:len(text)-1])
		if err == nil {
			n.isComplex = true
			n.complex128 = complex(0, f)
			n.simplifyComplex()
			return n, nil
		}
	}
	// Do integer test first so we get 0x123 etc.
	u, err := strconv.Btoui64(text, 0) // will fail for -0; fixed below.
	if err == nil {
		n.isUint = true
		n.uint64 = u
	}
	i, err := strconv.Btoi64(text, 0)
	if err == nil {
		n.isInt = true
		n.int64 = i
		if i == 0 {
			n.isUint = true // in case of -0.
			n.uint64 = u
		}
	}
	// If an integer extraction succeeded, promote the float.
	if n.isInt {
		n.isFloat = true
		n.float64 = float64(n.int64)
	} else if n.isUint {
		n.isFloat = true
		n.float64 = float64(n.uint64)
	} else {
		f, err := strconv.Atof64(text)
		if err == nil {
			n.isFloat = true
			n.float64 = f
			// If a floating-point extraction succeeded, extract the int if needed.
			if !n.isInt && float64(int64(f)) == f {
				n.isInt = true
				n.int64 = int64(f)
			}
			if !n.isUint && float64(uint64(f)) == f {
				n.isUint = true
				n.uint64 = uint64(f)
			}
		}
	}
	if !n.isInt && !n.isUint && !n.isFloat {
		return nil, fmt.Errorf("illegal number syntax: %q", text)
	}
	return n, nil
}

// simplifyComplex pulls out any other types that are represented by the complex number.
// These all require that the imaginary part be zero.
func (n *numberNode) simplifyComplex() {
	n.isFloat = imag(n.complex128) == 0
	if n.isFloat {
		n.float64 = real(n.complex128)
		n.isInt = float64(int64(n.float64)) == n.float64
		if n.isInt {
			n.int64 = int64(n.float64)
		}
		n.isUint = float64(uint64(n.float64)) == n.float64
		if n.isUint {
			n.uint64 = uint64(n.float64)
		}
	}
}

func (n *numberNode) String() string {
	return fmt.Sprintf("N=%s", n.text)
}

// stringNode holds a quoted string.
type stringNode struct {
	nodeType
	text string
}

func newString(text string) *stringNode {
	return &stringNode{nodeType: nodeString, text: text}
}

func (s *stringNode) String() string {
	return fmt.Sprintf("S=%#q", s.text)
}

// endNode represents an {{end}} action. It is represented by a nil pointer.
type endNode bool

func newEnd() *endNode {
	return nil
}

func (e *endNode) typ() nodeType {
	return nodeEnd
}

func (e *endNode) String() string {
	return "{{end}}"
}

// elseNode represents an {{else}} action.
type elseNode struct {
	nodeType
	line int
}

func newElse(line int) *elseNode {
	return &elseNode{nodeType: nodeElse, line: line}
}

func (e *elseNode) typ() nodeType {
	return nodeElse
}

func (e *elseNode) String() string {
	return "{{else}}"
}
// ifNode represents an {{if}} action and its commands.
// TODO: what should evaluation look like? is a pipeline enough?
type ifNode struct {
	nodeType
	line     int
	pipeline []*commandNode
	list     *listNode
	elseList *listNode
}

func newIf(line int, pipeline []*commandNode, list, elseList *listNode) *ifNode {
	return &ifNode{nodeType: nodeIf, line: line, pipeline: pipeline, list: list, elseList: elseList}
}

func (i *ifNode) String() string {
	if i.elseList != nil {
		return fmt.Sprintf("({{if %s}} %s {{else}} %s)", i.pipeline, i.list, i.elseList)
	}
	return fmt.Sprintf("({{if %s}} %s)", i.pipeline, i.list)
}

// rangeNode represents a {{range}} action and its commands.
type rangeNode struct {
	nodeType
	line     int
	pipeline []*commandNode
	list     *listNode
	elseList *listNode
}

func newRange(line int, pipeline []*commandNode, list, elseList *listNode) *rangeNode {
	return &rangeNode{nodeType: nodeRange, line: line, pipeline: pipeline, list: list, elseList: elseList}
}

func (r *rangeNode) String() string {
	if r.elseList != nil {
		return fmt.Sprintf("({{range %s}} %s {{else}} %s)", r.pipeline, r.list, r.elseList)
	}
	return fmt.Sprintf("({{range %s}} %s)", r.pipeline, r.list)
}

// templateNode represents a {{template}} action.
type templateNode struct {
	nodeType
	line     int
	name     node
	pipeline []*commandNode
}

func newTemplate(line int, name node, pipeline []*commandNode) *templateNode {
	return &templateNode{nodeType: nodeTemplate, line: line, name: name, pipeline: pipeline}
}

func (t *templateNode) String() string {
	return fmt.Sprintf("{{template %s %s}}", t.name, t.pipeline)
}

// withNode represents a {{with}} action and its commands.
type withNode struct {
	nodeType
	line     int
	pipeline []*commandNode
	list     *listNode
	elseList *listNode
}

func newWith(line int, pipeline []*commandNode, list, elseList *listNode) *withNode {
	return &withNode{nodeType: nodeWith, line: line, pipeline: pipeline, list: list, elseList: elseList}
}

func (w *withNode) String() string {
	if w.elseList != nil {
		return fmt.Sprintf("({{with %s}} %s {{else}} %s)", w.pipeline, w.list, w.elseList)
	}
	return fmt.Sprintf("({{with %s}} %s)", w.pipeline, w.list)
}


// Parsing.

// New allocates a new template with the given name.
func New(name string) *Template {
	return &Template{
		name:  name,
		funcs: make(map[string]reflect.Value),
	}
}

// Funcs adds to the template's function map the elements of the
// argument map.   It panics if a value in the map is not a function
// with appropriate return type.
// The return value is the template, so calls can be chained.
func (t *Template) Funcs(funcMap FuncMap) *Template {
	addFuncs(t.funcs, funcMap)
	return t
}

// errorf formats the error and terminates processing.
func (t *Template) errorf(format string, args ...interface{}) {
	t.root = nil
	format = fmt.Sprintf("template: %s:%d: %s", t.name, t.lex.lineNumber(), format)
	panic(fmt.Errorf(format, args...))
}

// error terminates processing.
func (t *Template) error(err os.Error) {
	t.errorf("%s", err)
}

// expect consumes the next token and guarantees it has the required type.
func (t *Template) expect(expected itemType, context string) item {
	token := t.next()
	if token.typ != expected {
		t.errorf("expected %s in %s; got %s", expected, context, token)
	}
	return token
}

// unexpected complains about the token and terminates processing.
func (t *Template) unexpected(token item, context string) {
	t.errorf("unexpected %s in %s", token, context)
}

// recover is the handler that turns panics into returns from the top
// level of Parse or Execute.
func (t *Template) recover(errp *os.Error) {
	e := recover()
	if e != nil {
		if _, ok := e.(runtime.Error); ok {
			panic(e)
		}
		t.stopParse()
		*errp = e.(os.Error)
	}
	return
}

// startParse starts the template parsing from the lexer.
func (t *Template) startParse(set *Set, lex *lexer) {
	t.root = nil
	t.set = set
	t.lex = lex
}

// stopParse terminates parsing.
func (t *Template) stopParse() {
	t.set, t.lex = nil, nil
}

// atEOF returns true if, possibly after spaces, we're at EOF.
func (t *Template) atEOF() bool {
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

// Parse parses the template definition string to construct an internal representation
// of the template for execution.
func (t *Template) Parse(s string) (err os.Error) {
	t.startParse(nil, lex(t.name, s))
	defer t.recover(&err)
	t.parse(true)
	t.stopParse()
	return
}

// ParseInSet parses the template definition string to construct an internal representation
// of the template for execution. Function bindings are checked against those in the set.
func (t *Template) ParseInSet(s string, set *Set) (err os.Error) {
	t.startParse(set, lex(t.name, s))
	defer t.recover(&err)
	t.parse(true)
	t.stopParse()
	return
}

// parse is the helper for Parse. It triggers an error if we expect EOF but don't reach it.
func (t *Template) parse(toEOF bool) (next node) {
	t.root, next = t.itemList(true)
	if toEOF && next != nil {
		t.errorf("unexpected %s", next)
	}
	return next
}

// itemList:
//	textOrAction*
// Terminates at EOF and at {{end}} or {{else}}, which is returned separately.
// The toEOF flag tells whether we expect to reach EOF.
func (t *Template) itemList(toEOF bool) (list *listNode, next node) {
	list = newList()
	for t.peek().typ != itemEOF {
		n := t.textOrAction()
		switch n.typ() {
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
func (t *Template) textOrAction() node {
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
func (t *Template) action() (n node) {
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
	return newAction(t.lex.lineNumber(), t.pipeline("command"))
}

// Pipeline:
//	field or command
//	pipeline "|" pipeline
func (t *Template) pipeline(context string) (pipe []*commandNode) {
	for {
		switch token := t.next(); token.typ {
		case itemRightDelim:
			if len(pipe) == 0 {
				t.errorf("missing value for %s", context)
			}
			return
		case itemBool, itemComplex, itemDot, itemField, itemIdentifier, itemNumber, itemRawString, itemString:
			t.backup()
			pipe = append(pipe, t.command())
		default:
			t.unexpected(token, context)
		}
	}
	return
}

func (t *Template) parseControl(context string) (lineNum int, pipe []*commandNode, list, elseList *listNode) {
	lineNum = t.lex.lineNumber()
	pipe = t.pipeline(context)
	var next node
	list, next = t.itemList(false)
	switch next.typ() {
	case nodeEnd: //done
	case nodeElse:
		elseList, next = t.itemList(false)
		if next.typ() != nodeEnd {
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
func (t *Template) ifControl() node {
	return newIf(t.parseControl("if"))
}

// Range:
//	{{range pipeline}} itemList {{end}}
//	{{range pipeline}} itemList {{else}} itemList {{end}}
// Range keyword is past.
func (t *Template) rangeControl() node {
	return newRange(t.parseControl("range"))
}

// With:
//	{{with pipeline}} itemList {{end}}
//	{{with pipeline}} itemList {{else}} itemList {{end}}
// If keyword is past.
func (t *Template) withControl() node {
	return newWith(t.parseControl("with"))
}


// End:
//	{{end}}
// End keyword is past.
func (t *Template) endControl() node {
	t.expect(itemRightDelim, "end")
	return newEnd()
}

// Else:
//	{{else}}
// Else keyword is past.
func (t *Template) elseControl() node {
	t.expect(itemRightDelim, "else")
	return newElse(t.lex.lineNumber())
}

// Template:
//	{{template stringValue pipeline}}
// Template keyword is past.  The name must be something that can evaluate
// to a string.
func (t *Template) templateControl() node {
	var name node
	switch token := t.next(); token.typ {
	case itemIdentifier:
		if _, ok := findFunction(token.val, t, t.set); !ok {
			t.errorf("function %q not defined", token.val)
		}
		name = newIdentifier(token.val)
	case itemDot:
		name = newDot()
	case itemField:
		name = newField(token.val)
	case itemString, itemRawString:
		s, err := strconv.Unquote(token.val)
		if err != nil {
			t.error(err)
		}
		name = newString(s)
	default:
		t.unexpected(token, "template invocation")
	}
	pipeline := t.pipeline("template")
	return newTemplate(t.lex.lineNumber(), name, pipeline)
}

// command:
// space-separated arguments up to a pipeline character or right delimiter.
// we consume the pipe character but leave the right delim to terminate the action.
func (t *Template) command() *commandNode {
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
			if _, ok := findFunction(token.val, t, t.set); !ok {
				t.errorf("function %q not defined", token.val)
			}
			cmd.append(newIdentifier(token.val))
		case itemDot:
			cmd.append(newDot())
		case itemField:
			cmd.append(newField(token.val))
		case itemBool:
			cmd.append(newBool(token.val == "true"))
		case itemComplex, itemNumber:
			number, err := newNumber(token.val, token.typ == itemComplex)
			if err != nil {
				t.error(err)
			}
			cmd.append(number)
		case itemString, itemRawString:
			s, err := strconv.Unquote(token.val)
			if err != nil {
				t.error(err)
			}
			cmd.append(newString(s))
		default:
			t.unexpected(token, "command")
		}
	}
	if len(cmd.args) == 0 {
		t.errorf("empty command")
	}
	return cmd
}
