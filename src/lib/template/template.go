// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Template library.  See http://code.google.com/p/json-template/wiki/Reference
// TODO: document this here as well.
package template

import (
	"fmt";
	"io";
	"os";
	"reflect";
	"strings";
	"template";
)

var ErrLBrace = os.NewError("unexpected opening brace")
var ErrUnmatchedRBrace = os.NewError("unmatched closing brace")
var ErrUnmatchedLBrace = os.NewError("unmatched opening brace")
var ErrBadDirective = os.NewError("unrecognized directive name")
var ErrEmptyDirective = os.NewError("empty directive")
var ErrFields = os.NewError("incorrect fields for directive")
var ErrSyntax = os.NewError("directive out of place")
var ErrNoEnd = os.NewError("section does not have .end")
var ErrNoVar = os.NewError("variable name not in struct");
var ErrBadType = os.NewError("unsupported type for variable");
var ErrNotStruct = os.NewError("driver must be a struct")
var ErrNoFormatter = os.NewError("unknown formatter")

// All the literals are aces.
var lbrace = []byte{ '{' }
var rbrace = []byte{ '}' }
var space = []byte{ ' ' }

// The various types of "tokens", which are plain text or brace-delimited descriptors
const (
	Alternates = iota;
	Comment;
	End;
	Literal;
	Or;
	Repeated;
	Section;
	Text;
	Variable;
)

// FormatterMap is the type describing the mapping from formatter
// names to the functions that implement them.
type FormatterMap map[string] func(reflect.Value) string

// Built-in formatters.
var builtins = FormatterMap {
	"html" : HtmlFormatter,
	"str" : StringFormatter,
	"" : StringFormatter,
}

type template struct {
	errorchan	chan *os.Error;	// for erroring out
	linenum	*int;	// shared by all templates derived from this one
	parent	*template;
	data	reflect.Value;	// the driver data for this section etc.
	fmap	FormatterMap;	// formatters for variables
	buf	[]byte;	// input text to process
	p	int;	// position in buf
	wr	io.Write;	// where to send output
}

// Create a top-level template
func newTemplate(ch chan *os.Error, linenum *int, buf []byte, data reflect.Value, fmap FormatterMap, wr io.Write) *template {
	t := new(template);
	t.errorchan = ch;
	t.linenum = linenum;
	*linenum = 1;
	t.parent = nil;
	t.data = data;
	t.buf = buf;
	t.p = 0;
	t.fmap = fmap;
	t.wr = wr;
	return t;
}

// Create a template deriving from its parent
func childTemplate(parent *template, buf []byte, data reflect.Value) *template {
	t := newTemplate(parent.errorchan, parent.linenum, buf, data, parent.fmap, parent.wr);
	t.parent = parent;
	return t;
}

// Report error and stop generation.
func (t *template) error(err *os.Error, args ...) {
	fmt.Fprintf(os.Stderr, "template error: line %d: %s%s\n", *t.linenum, err, fmt.Sprint(args));  // TODO: drop this? (only way to get line number)
	t.errorchan <- err;
	sys.Goexit();
}

func white(c uint8) bool {
	return c == ' ' || c == '\t' || c == '\n'
}

func (t *template) execute()
func (t *template) executeSection(w []string)

// nextItem returns the next item from the input buffer.  If the returned
// item is empty, we are at EOF.  The item will be either a brace-
// delimited string or a non-empty string between brace-delimited
// strings.  Most tokens stop at (but include, if plain text) a newline.
// Action tokens on a line by themselves drop the white space on
// either side, up to and including the newline.
func (t *template) nextItem() []byte {
	brace := false;	// are we waiting for an opening brace?
	special := false;	// is this a {.foo} directive, which means trim white space?
	// Delete surrounding white space if this {.foo} is the only thing on the line.
	trim_white := t.p == 0 || t.buf[t.p-1] == '\n';
	only_white := true;	// we have seen only white space so far
	var i int;
	start := t.p;
Loop:
	for i = t.p; i < len(t.buf); i++ {
		switch t.buf[i] {
		case '\n':
			*t.linenum++;
			i++;
			break Loop;
		case ' ', '\t':
			// white space, do nothing
		case '{':
			if brace {
				t.error(ErrLBrace)
			}
			// anything interesting already on the line?
			if !only_white {
				break Loop;
			}
			// is it a directive or comment?
			if i+2 < len(t.buf) && (t.buf[i+1] == '.' || t.buf[i+1] == '#') {
				special = true;
				if trim_white && only_white {
					start = i;
				}
			} else if i > t.p+1 {  // have some text accumulated so stop before '{'
				break Loop;
			}
			brace = true;
		case '}':
			if !brace {
				t.error(ErrUnmatchedRBrace)
			}
			brace = false;
			i++;
			break Loop;
		default:
			only_white = false;
		}
	}
	if brace {
		t.error(ErrUnmatchedLBrace)
	}
	item := t.buf[start:i];
	if special && trim_white {
		// consume trailing white space
		for ; i < len(t.buf) && white(t.buf[i]); i++ {
			if t.buf[i] == '\n' {
				i++;
				break	// stop after newline
			}
		}
	}
	t.p = i;
	return item
}

// Turn a byte array into a white-space-split array of strings.
func words(buf []byte) []string {
	s := make([]string, 0, 5);
	p := 0; // position in buf
	// one word per loop
	for i := 0; ; i++ {
		// skip white space
		for ; p < len(buf) && white(buf[p]); p++ {
		}
		// grab word
		start := p;
		for ; p < len(buf) && !white(buf[p]); p++ {
		}
		if start == p {	// no text left
			break
		}
		if i == cap(s) {
			ns := make([]string, 2*cap(s));
			for j := range s {
				ns[j] = s[j]
			}
			s = ns;
		}
		s = s[0:i+1];
		s[i] = string(buf[start:p])
	}
	return s
}

// Analyze an item and return its type and, if it's an action item, an array of
// its constituent words.
func (t *template) analyze(item []byte) (tok int, w []string) {
	// item is known to be non-empty
	if item[0] != '{' {
		tok = Text;
		return
	}
	if item[len(item)-1] != '}' {
		t.error(ErrUnmatchedLBrace)  // should not happen anyway
	}
	if len(item) <= 2 {
		t.error(ErrEmptyDirective)
	}
	// Comment
	if item[1] == '#' {
		tok = Comment;
		return
	}
	// Split into words
	w = words(item[1: len(item)-1]);  // drop final brace
	if len(w) == 0 {
		t.error(ErrBadDirective)
	}
	if len(w[0]) == 0 {
		t.error(ErrEmptyDirective)
	}
	if len(w) == 1 && w[0][0] != '.' {
		tok = Variable;
		return;
	}
	switch w[0] {
	case ".meta-left", ".meta-right", ".space":
		tok = Literal;
		return;
	case ".or":
		tok = Or;
		return;
	case ".end":
		tok = End;
		return;
	case ".section":
		if len(w) != 2 {
			t.error(ErrFields, ": ", string(item))
		}
		tok = Section;
		return;
	case ".repeated":
		if len(w) != 3 || w[1] != "section" {
			t.error(ErrFields, ": ", string(item))
		}
		tok = Repeated;
		return;
	case ".alternates":
		if len(w) != 2 || w[1] != "with" {
			t.error(ErrFields, ": ", string(item))
		}
		tok = Alternates;
		return;
	}
	t.error(ErrBadDirective, ": ", string(item));
	return
}

// If the data for this template is a struct, find the named variable.
func (t *template) findVar(s string) (int, int) {
	typ, ok := t.data.Type().(reflect.StructType);
	if ok {
		for i := 0; i < typ.Len(); i++ {
			name, ftyp, tag, offset := typ.Field(i);
			if name == s {
				return i, ftyp.Kind()
			}
		}
	}
	return -1, -1
}

// Is there no data to look at?
func empty(v reflect.Value, indirect_ok bool) bool {
	v = reflect.Indirect(v);
	if v == nil {
		return true
	}
	switch v.Type().Kind() {
	case reflect.StructKind:
		return false;
	case reflect.ArrayKind:
		return v.(reflect.ArrayValue).Len() == 0;
	}
	return true;
}

// Execute a ".repeated" section
func (t *template) executeRepeated(w []string) {
	if w[1] != "section" {
		t.error(ErrSyntax, `: .repeated must have "section"`)
	}
	// Find driver array/struct for this section.  It must be in the current struct.
	// The special name "@" leaves us at this level.
	var field reflect.Value;
	if w[2] == "@" {
		field = t.data
	} else {
		i, kind := t.findVar(w[1]);
		if i < 0 {
			t.error(ErrNoVar, ": ", w[2]);
		}
		field = reflect.Indirect(t.data.(reflect.StructValue).Field(i));
	}
	// Must be an array/slice
	if field != nil && field.Kind() != reflect.ArrayKind {
		t.error(ErrBadType, " in .repeated: ", w[2], " ", field.Type().String());
	}
	// Scan repeated section, remembering slice of text we must execute.
	nesting := 0;
	start := t.p;
	end := t.p;
Loop:
	for {
		item := t.nextItem();
		if len(item) ==  0 {
			t.error(ErrNoEnd)
		}
		tok, s := t.analyze(item);
		switch tok {
		case Comment:
			continue;	// just ignore it
		case End:
			if nesting == 0 {
				break Loop
			}
			nesting--;
		case Repeated, Section:
			nesting++;
		case Literal, Or, Text, Variable:
			// just accumulate
		default:
			panic("unknown section item", string(item));
		}
		end = t.p
	}
	if field != nil {
		array := field.(reflect.ArrayValue);
		for i := 0; i < array.Len(); i++ {
			elem := reflect.Indirect(array.Elem(i));
			tmp := childTemplate(t, t.buf[start:end], elem);
			tmp.execute();
		}
	}
}

// Execute a ".section"
func (t *template) executeSection(w []string) {
	// Find driver array/struct for this section.  It must be in the current struct.
	// The special name "@" leaves us at this level.
	var field reflect.Value;
	if w[1] == "@" {
		field = t.data
	} else {
		i, kind := t.findVar(w[1]);
		if i < 0 {
			t.error(ErrNoVar, ": ", w[1]);
		}
		field = t.data.(reflect.StructValue).Field(i);
	}
	// Scan section, remembering slice of text we must execute.
	orFound := false;
	nesting := 0;  // How deeply are .section and .repeated nested?
	start := t.p;
	end := t.p;
	accumulate := !empty(field, true);	// Keep this section if there's data
Loop:
	for {
		item := t.nextItem();
		if len(item) ==  0 {
			t.error(ErrNoEnd)
		}
		tok, s := t.analyze(item);
		switch tok {
		case Comment:
			continue;	// just ignore it
		case End:
			if nesting == 0 {
				break Loop
			}
			nesting--;
		case Or:
			if nesting > 0 {	// just accumulate
				break
			}
			if orFound {
				t.error(ErrSyntax, ": .or");
			}
			orFound = true;
			if !accumulate {
				// No data; execute the .or instead
				start = t.p;
				end = t.p;
				accumulate = true;
				continue;
			} else {
				// Data present so disregard the .or section
				accumulate = false
			}
		case Repeated, Section:
			nesting++;
		case Literal, Text, Variable:
			// just accumulate
		default:
			panic("unknown section item", string(item));
		}
		if accumulate {
			end = t.p
		}
	}
	tmp := childTemplate(t, t.buf[start:end], field);
	tmp.execute();
}

// Look up a variable, up through the parent if necessary.
func (t *template) varValue(name string) reflect.Value {
	i, kind := t.findVar(name);
	if i < 0 {
		if t.parent == nil {
			t.error(ErrNoVar, ": ", name)
		}
		return t.parent.varValue(name);
	}
	return t.data.(reflect.StructValue).Field(i);
}

// Evalute a variable, looking up through the parent if necessary.
// If it has a formatter attached ({var|formatter}) run that too.
func (t *template) evalVariable(name_formatter string) string {
	name := name_formatter;
	formatter := "";
	bar := strings.Index(name_formatter, "|");
	if bar >= 0 {
		name = name_formatter[0:bar];
		formatter = name_formatter[bar+1:len(name_formatter)];
	}
	val := t.varValue(name);
	// is it in user-supplied map?
	if fn, ok := t.fmap[formatter]; ok {
		return fn(val)
	}
	// is it in builtin map?
	if fn, ok := builtins[formatter]; ok {
		return fn(val)
	}
	t.error(ErrNoFormatter, ": ", formatter);
	panic("notreached");
}

func (t *template) execute() {
	for {
		item := t.nextItem();
		if len(item) == 0 {
			return
		}
		tok, w := t.analyze(item);
		switch tok {
		case Comment:
			break;
		case Text:
			t.wr.Write(item);
		case Literal:
			switch w[0] {
			case ".meta-left":
				t.wr.Write(lbrace);
			case ".meta-right":
				t.wr.Write(rbrace);
			case ".space":
				t.wr.Write(space);
			default:
				panic("unknown literal: ", w[0]);
			}
		case Variable:
			t.wr.Write(io.StringBytes(t.evalVariable(w[0])));
		case Or, End, Alternates:
			t.error(ErrSyntax, ": ", string(item));
		case Section:
			t.executeSection(w);
		case Repeated:
			t.executeRepeated(w);
		default:
			panic("bad directive in execute:", string(item));
		}
	}
}

func Execute(s string, data interface{}, fmap FormatterMap, wr io.Write) *os.Error {
	// Extract the driver struct.
	val := reflect.Indirect(reflect.NewValue(data));
	sval, ok1 := val.(reflect.StructValue);
	if !ok1 {
		return ErrNotStruct
	}
	ch := make(chan *os.Error);
	var linenum int;
	t := newTemplate(ch, &linenum, io.StringBytes(s), val, fmap, wr);
	go func() {
		t.execute();
		ch <- nil;	// clean return;
	}();
	return <-ch;
}
