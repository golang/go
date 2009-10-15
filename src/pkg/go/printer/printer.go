// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The printer package implements printing of AST nodes.
package printer

import (
	"bytes";
	"container/vector";
	"fmt";
	"go/ast";
	"go/token";
	"io";
	"os";
	"reflect";
	"runtime";
	"strings";
	"tabwriter";
)


const (
	debug = false;  // enable for debugging
	maxNewlines = 3;  // maximum vertical white space
)


// Printing is controlled with these flags supplied
// to Fprint via the mode parameter.
//
const (
	GenHTML uint = 1 << iota;  // generate HTML
	RawFormat;  // do not use a tabwriter; if set, UseSpaces is ignored
	UseSpaces;  // use spaces instead of tabs for indentation and alignment
)


type whiteSpace int

const (
	ignore = whiteSpace(0);
	blank = whiteSpace(' ');
	vtab = whiteSpace('\v');
	newline = whiteSpace('\n');
	formfeed = whiteSpace('\f');
	indent = whiteSpace('>');
	unindent = whiteSpace('<');
)


var (
	esc = []byte{tabwriter.Escape};
	htab = []byte{'\t'};
	htabs = [...]byte{'\t', '\t', '\t', '\t', '\t', '\t', '\t', '\t'};
	newlines = [...]byte{'\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n'};  // more than maxNewlines
	ampersand = strings.Bytes("&amp;");
	lessthan = strings.Bytes("&lt;");
	greaterthan = strings.Bytes("&gt;");
)


// A lineTag is a token.Position that is used to print
// line tag id's of the form "L%d" where %d stands for
// the line indicated by position.
//
type lineTag token.Position


// A htmlTag specifies a start and end tag.
type htmlTag struct {
	start, end string;  // empty if tags are absent
}


type printer struct {
	// Configuration (does not change after initialization)
	output io.Writer;
	mode uint;
	errors chan os.Error;

	// Current state
	written int;  // number of bytes written
	indent int;  // current indentation
	escape bool;  // true if in escape sequence

	// Buffered whitespace
	buffer []whiteSpace;

	// The (possibly estimated) position in the generated output;
	// in AST space (i.e., pos is set whenever a token position is
	// known accurately, and updated dependending on what has been
	// written)
	pos token.Position;

	// The value of pos immediately after the last item has been
	// written using writeItem.
	last token.Position;

	// HTML support
	tag htmlTag;  // tag to be used around next item
	lastTaggedLine int;  // last line for which a line tag was written

	// The list of comments; or nil.
	comment *ast.CommentGroup;
}


func (p *printer) init(output io.Writer, mode uint) {
	p.output = output;
	p.mode = mode;
	p.errors = make(chan os.Error);
	p.buffer = make([]whiteSpace, 0, 16);  // whitespace sequences are short
}


// write0 writes raw (uninterpreted) data to p.output and handles errors.
// write0 does not indent after newlines, and does not HTML-escape or update p.pos.
//
func (p *printer) write0(data []byte) {
	n, err := p.output.Write(data);
	p.written += n;
	if err != nil {
		p.errors <- err;
		runtime.Goexit();
	}
}


// write interprets data and writes it to p.output. It inserts indentation
// after a line break unless in a tabwriter escape sequence, and it HTML-
// escapes characters if GenHTML is set. It updates p.pos as a side-effect.
//
func (p *printer) write(data []byte) {
	i0 := 0;
	for i, b := range data {
		switch b {
		case '\n', '\f':
			// write segment ending in b
			p.write0(data[i0 : i+1]);

			// update p.pos
			p.pos.Offset += i+1 - i0;
			p.pos.Line++;
			p.pos.Column = 1;

			if !p.escape {
				// write indentation
				// use "hard" htabs - indentation columns
				// must not be discarded by the tabwriter
				j := p.indent;
				for ; j > len(htabs); j -= len(htabs) {
					p.write0(&htabs);
				}
				p.write0(htabs[0:j]);

				// update p.pos
				p.pos.Offset += p.indent;
				p.pos.Column += p.indent;
			}

			// next segment start
			i0 = i+1;

		case '&', '<', '>':
			if p.mode & GenHTML != 0 {
				// write segment ending in b
				p.write0(data[i0 : i]);

				// write HTML-escaped b
				var esc []byte;
				switch b {
				case '&': esc = ampersand;
				case '<': esc = lessthan;
				case '>': esc = greaterthan;
				}
				p.write0(esc);

				// update p.pos
				d := i+1 - i0;
				p.pos.Offset += d;
				p.pos.Column += d;

				// next segment start
				i0 = i+1;
			}

		case tabwriter.Escape:
			p.escape = !p.escape;
		}
	}

	// write remaining segment
	p.write0(data[i0 : len(data)]);

	// update p.pos
	d := len(data) - i0;
	p.pos.Offset += d;
	p.pos.Column += d;
}


func (p *printer) writeNewlines(n int) {
	if n > 0 {
		if n > maxNewlines {
			n = maxNewlines;
		}
		p.write(newlines[0 : n]);
	}
}


// writeItem writes data at position pos. data is the text corresponding to
// a single lexical token, but may also be comment text. pos is the actual
// (or at least very accurately estimated) position of the data in the original
// source text. The data may be tagged, depending on p.mode and the setLineTag
// parameter. writeItem updates p.last to the position immediately following
// the data.
//
func (p *printer) writeItem(pos token.Position, data []byte, setLineTag bool) {
	p.pos = pos;
	if debug {
		// do not update p.pos - use write0
		p.write0(strings.Bytes(fmt.Sprintf("[%d:%d]", pos.Line, pos.Column)));
	}
	if p.mode & GenHTML != 0 {
		// no html-escaping and no p.pos update for tags - use write0
		if setLineTag && pos.Line > p.lastTaggedLine {
			// id's must be unique within a document: set
			// line tag only if line number has increased
			// (note: for now write complete start and end
			// tag - shorter versions seem to have issues
			// with Safari)
			p.tag.start = fmt.Sprintf(`<a id="L%d"></a>`, pos.Line);
			p.lastTaggedLine = pos.Line;
		}
		// write start tag, if any
		if p.tag.start != "" {
			p.write0(strings.Bytes(p.tag.start));
			p.tag.start = "";  // tag consumed
		}
		p.write(data);
		// write end tag, if any
		if p.tag.end != "" {
			p.write0(strings.Bytes(p.tag.end));
			p.tag.end = "";  // tag consumed
		}
	} else {
		p.write(data);
	}
	p.last = p.pos;
}


// writeCommentPrefix writes the whitespace before a comment.
// If there is any pending whitespace, it consumes as much of
// it as is likely to help the comment position properly.
// line is the comment line, isFirst indicates if this is the
// first comment in a group of comments.
//
func (p *printer) writeCommentPrefix(line int, isFirst bool) {
	if !p.last.IsValid() {
		// there was no preceeding item and the comment is the
		// first item to be printed - don't write any whitespace
		return;
	}

	n := line - p.last.Line;
	if n == 0 {
		// comment on the same line as last item:
		// separate with at least one tab
		hasTab := false;
		if isFirst {
			j := 0;
			for i, ch := range p.buffer {
				switch ch {
				case blank:
					// ignore any blanks before a comment
					p.buffer[i] = ignore;
					continue;
				case vtab:
					// respect existing tabs - important
					// for proper formatting of commented structs
					hasTab = true;
					continue;
				case indent:
					// apply pending indentation
					continue;
				}
				j = i;
				break;
			}
			p.writeWhitespace(j);
		}
		// make sure there is at least one tab
		if !hasTab {
			p.write(htab);
		}

	} else {
		// comment on a different line:
		// separate with at least one line break
		if isFirst {
			j := 0;
			for i, ch := range p.buffer {
				switch ch {
				case blank, vtab:
					// ignore any horizontal whitespace before line breaks
					p.buffer[i] = ignore;
					continue;
				case indent:
					// apply pending indentation
					continue;
				case newline, formfeed:
					// TODO(gri): may want to keep formfeed info in some cases
					p.buffer[i] = ignore;
				}
				j = i;
				break;
			}
			p.writeWhitespace(j);
		}
		p.writeNewlines(n);
	}
}


// Collapse contiguous sequences of '\t' in a []byte to a single '\t'.
func collapseTabs(src []byte) []byte {
	dst := make([]byte, len(src));
	j := 0;
	for i, c := range src {
		if c != '\t' || i == 0 || src[i-1] != '\t' {
			dst[j] = c;
			j++;
		}
	}
	return dst[0 : j];
}


func (p *printer) writeComment(comment *ast.Comment) {
	// If there are tabs in the comment text, they were probably introduced
	// to align the comment contents. If the same tab settings were used as
	// by the printer, reducing tab sequences to single tabs will yield the
	// original comment again after reformatting via the tabwriter.
	text := comment.Text;
	if p.mode & RawFormat == 0 {
		// tabwriter is used
		text = collapseTabs(comment.Text);
	}

	// write comment
	p.writeItem(comment.Pos(), text, false);
}


// writeCommentSuffix writes a line break after a comment if indicated
// and processes any leftover indentation information. If a line break
// is needed, the kind of break (newline vs formfeed) depends on the
// pending whitespace.
//
func (p *printer) writeCommentSuffix(needsLinebreak bool) {
	for i, ch := range p.buffer {
		switch ch {
		case blank, vtab:
			// ignore trailing whitespace
			p.buffer[i] = ignore;
		case indent, unindent:
			// don't loose indentation information
		case newline, formfeed:
			// if we need a line break, keep exactly one
			if needsLinebreak {
				needsLinebreak = false;
			} else {
				p.buffer[i] = ignore;
			}
		}
	}
	p.writeWhitespace(len(p.buffer));

	// make sure we have a line break
	if needsLinebreak {
		p.write([]byte{'\n'});
	}
}


// intersperseComments consumes all comments that appear before the next token
// and prints it together with the buffered whitespace (i.e., the whitespace
// that needs to be written before the next token). A heuristic is used to mix
// the comments and whitespace.
//
func (p *printer) intersperseComments(next token.Position) {
	isFirst := true;
	needsLinebreak := false;
	for ; p.commentBefore(next); p.comment = p.comment.Next {
		for _, c := range p.comment.List {
			p.writeCommentPrefix(c.Pos().Line, isFirst);
			isFirst = false;
			p.writeComment(c);
			needsLinebreak = c.Text[1] == '/';
		}
	}
	p.writeCommentSuffix(needsLinebreak);
}


// whiteWhitespace writes the first n whitespace entries.
func (p *printer) writeWhitespace(n int) {
	// write entries
	var data [1]byte;
	for i := 0; i < n; i++ {
		switch ch := p.buffer[i]; ch {
		case ignore:
			// ignore!
		case indent:
			p.indent++;
		case unindent:
			p.indent--;
			if p.indent < 0 {
				// handle gracefully unless in debug mode
				if debug {
					panicln("negative indentation:", p.indent);
				}
				p.indent = 0;
			}
		case newline, formfeed:
			// A line break immediately followed by a "correcting"
			// unindent is swapped with the unindent - this permits
			// proper label positioning. If a comment is between
			// the line break and the label, the unindent is not
			// part of the comment whitespace prefix and the comment
			// will be positioned correctly indented.
			if i+1 < n && p.buffer[i+1] == unindent {
				p.buffer[i], p.buffer[i+1] = unindent, ch;
				i--;  // do it again
				continue;
			}
			fallthrough;
		default:
			data[0] = byte(ch);
			p.write(&data);
		}
	}

	// shift remaining entries down
	i := 0;
	for ; n < len(p.buffer); n++ {
		p.buffer[i] = p.buffer[n];
		i++;
	}
	p.buffer = p.buffer[0:i];
}


// ----------------------------------------------------------------------------
// Printing interface

// print prints a list of "items" (roughly corresponding to syntactic
// tokens, but also including whitespace and formatting information).
// It is the only print function that should be called directly from
// any of the AST printing functions below.
//
// Whitespace is accumulated until a non-whitespace token appears. Any
// comments that need to appear before that token are printed first,
// taking into account the amount and structure of any pending white-
// space for best comment placement. Then, any leftover whitespace is
// printed, followed by the actual token.
//
func (p *printer) print(args ...) {
	setLineTag := false;
	v := reflect.NewValue(args).(*reflect.StructValue);
	for i := 0; i < v.NumField(); i++ {
		f := v.Field(i);

		next := p.pos;  // estimated position of next item
		var data []byte;
		switch x := f.Interface().(type) {
		case whiteSpace:
			i := len(p.buffer);
			if i == cap(p.buffer) {
				// Whitespace sequences are very short so this should
				// never happen. Handle gracefully (but possibly with
				// bad comment placement) if it does happen.
				p.writeWhitespace(i);
				i = 0;
			}
			p.buffer = p.buffer[0 : i+1];
			p.buffer[i] = x;
		case []byte:
			data = x;
		case string:
			data = strings.Bytes(x);
		case token.Token:
			data = strings.Bytes(x.String());
		case token.Position:
			if x.IsValid() {
				next = x;  // accurate position of next item
			}
		case lineTag:
			pos := token.Position(x);
			if pos.IsValid() {
				next = pos;  // accurate position of next item
				setLineTag = true;
			}
		case htmlTag:
			p.tag = x;  // tag surrounding next item
		default:
			panicln("print: unsupported argument type", f.Type().String());
		}
		p.pos = next;

		if data != nil {
			p.flush(next);

			// intersperse extra newlines if present in the source
			// (don't do this in flush as it will cause extra newlines
			// at the end of a file)
			p.writeNewlines(next.Line - p.pos.Line);

			p.writeItem(next, data, setLineTag);
			setLineTag = false;
		}
	}
}


// commentBefore returns true iff the current comment occurs
// before the next position in the source code.
//
func (p *printer) commentBefore(next token.Position) bool {
	return p.comment != nil && p.comment.List[0].Pos().Offset < next.Offset;
}


// Flush prints any pending comments and whitespace occuring
// textually before the position of the next item.
//
func (p *printer) flush(next token.Position) {
	// if there are comments before the next item, intersperse them
	if p.commentBefore(next) {
		p.intersperseComments(next);
	}
	// write any leftover whitespace
	p.writeWhitespace(len(p.buffer));
}


// ----------------------------------------------------------------------------
// Printing of common AST nodes.


// Print as many newlines as necessary (but at least min and and at most
// max newlines) to get to the current line. If newSection is set, the
// first newline is printed as a formfeed. Returns true if any line break
// was printed; returns false otherwise.
//
// TODO(gri): Reconsider signature (provide position instead of line)
//
func (p *printer) linebreak(line, min, max int, newSection bool) (printedBreak bool) {
	n := line - p.pos.Line;
	switch {
	case n < min: n = min;
	case n > max: n = max;
	}
	if n > 0 && newSection {
		p.print(formfeed);
		n--;
		printedBreak = true;
	}
	for ; n > 0; n-- {
		p.print(newline);
		printedBreak = true;
	}
	return;
}


// TODO(gri): The code for printing lead and line comments
//            should be eliminated in favor of reusing the
//            comment intersperse mechanism above somehow.

// Print a list of individual comments.
func (p *printer) commentList(list []*ast.Comment) {
	for i, c := range list {
		t := c.Text;
		p.print(c.Pos(), t);
		if t[1] == '/' && i+1 < len(list) {
			//-style comment which is not at the end; print a newline
			p.print(newline);
		}
	}
}


// Print a lead comment followed by a newline.
func (p *printer) leadComment(d *ast.CommentGroup) {
	// Ignore the comment if we have comments interspersed (p.comment != nil).
	if p.comment == nil && d != nil {
		p.commentList(d.List);
		p.print(newline);
	}
}


// Print a tab followed by a line comment.
// A newline must be printed afterwards since
// the comment may be a //-style comment.
func (p *printer) lineComment(d *ast.CommentGroup) {
	// Ignore the comment if we have comments interspersed (p.comment != nil).
	if p.comment == nil && d != nil {
		p.print(vtab);
		p.commentList(d.List);
	}
}


func (p *printer) identList(list []*ast.Ident) {
	// convert into an expression list
	xlist := make([]ast.Expr, len(list));
	for i, x := range list {
		xlist[i] = x;
	}
	p.exprList(xlist, commaSep);
}


func (p *printer) stringList(list []*ast.BasicLit) {
	// convert into an expression list
	xlist := make([]ast.Expr, len(list));
	for i, x := range list {
		xlist[i] = x;
	}
	p.exprList(xlist, noIndent);
}


type exprListMode uint;
const (
	blankStart exprListMode = 1 << iota;  // print a blank before the list
	commaSep;  // elements are separated by commas
	commaTerm;  // elements are terminated by comma
	noIndent;  // no extra indentation in multi-line lists
)


// Print a list of expressions. If the list spans multiple
// source lines, the original line breaks are respected.
func (p *printer) exprList(list []ast.Expr, mode exprListMode) {
	if len(list) == 0 {
		return;
	}

	if mode & blankStart != 0 {
		p.print(blank);
	}

	if list[0].Pos().Line == list[len(list)-1].Pos().Line {
		// all list entries on a single line
		for i, x := range list {
			if i > 0 {
				if mode & commaSep != 0 {
					p.print(token.COMMA);
				}
				p.print(blank);
			}
			p.expr(x);
		}
		return;
	}

	// list entries span multiple lines;
	// use source code positions to guide line breaks
	line := list[0].Pos().Line;
	// don't add extra indentation if noIndent is set;
	// i.e., pretend that the first line is already indented
	indented := mode&noIndent != 0;
	// there may or may not be a line break before the first list
	// element; in any case indent once after the first line break
	if p.linebreak(line, 0, 2, true) && !indented {
		p.print(htab, indent);  // indent applies to next line
		indented = true;
	}
	for i, x := range list {
		prev := line;
		line = x.Pos().Line;
		if i > 0 {
			if mode & commaSep != 0 {
				p.print(token.COMMA);
			}
			if prev < line {
				// at least one line break, but respect an extra empty line
				// in the source
				if p.linebreak(x.Pos().Line, 1, 2, true) && !indented {
					p.print(htab, indent);  // indent applies to next line
					indented = true;
				}
			} else {
				p.print(blank);
			}
		}
		p.expr(x);
	}
	if mode & commaTerm != 0 {
		p.print(token.COMMA);
		if indented && mode&noIndent == 0 {
			// should always be indented here since we have a multi-line
			// expression list - be conservative and check anyway
			p.print(unindent);
		}
		p.print(formfeed);  // terminating comma needs a line break to look good
	} else if indented && mode&noIndent == 0 {
		p.print(unindent);
	}
}


func (p *printer) parameters(list []*ast.Field) {
	p.print(token.LPAREN);
	if len(list) > 0 {
		for i, par := range list {
			if i > 0 {
				p.print(token.COMMA, blank);
			}
			if len(par.Names) > 0 {
				p.identList(par.Names);
				p.print(blank);
			}
			p.expr(par.Type);
		}
	}
	p.print(token.RPAREN);
}


// Returns true if a separating semicolon is optional.
func (p *printer) signature(params, result []*ast.Field) (optSemi bool) {
	p.parameters(params);
	if result != nil {
		p.print(blank);

		if len(result) == 1 && result[0].Names == nil {
			// single anonymous result; no ()'s unless it's a function type
			f := result[0];
			if _, isFtyp := f.Type.(*ast.FuncType); !isFtyp {
				optSemi = p.expr(f.Type);
				return;
			}
		}

		p.parameters(result);
	}
	return;
}


func (p *printer) fieldList(lbrace token.Position, list []*ast.Field, rbrace token.Position, isIncomplete, isStruct bool) {
	if len(list) == 0 && !isIncomplete && !p.commentBefore(rbrace) {
		// no blank between keyword and {} in this case
		p.print(lbrace, token.LBRACE, rbrace, token.RBRACE);
		return;
	}

	// at least one entry or incomplete
	p.print(blank, lbrace, token.LBRACE, indent, formfeed);
	if isStruct {

		sep := vtab;
		if len(list) == 1 {
			sep = blank;
		}
		for i, f := range list {
			extraTabs := 0;
			p.leadComment(f.Doc);
			if len(f.Names) > 0 {
				p.identList(f.Names);
				p.print(sep);
				p.expr(f.Type);
				extraTabs = 1;
			} else {
				p.expr(f.Type);
				extraTabs = 2;
			}
			if f.Tag != nil {
				if len(f.Names) > 0 && sep == vtab {
					p.print(sep);
				}
				p.print(sep);
				p.expr(&ast.StringList{f.Tag});
				extraTabs = 0;
			}
			p.print(token.SEMICOLON);
			if f.Comment != nil {
				for ; extraTabs > 0; extraTabs-- {
					p.print(vtab);
				}
				p.lineComment(f.Comment);
			}
			if i+1 < len(list) || isIncomplete {
				p.print(newline);
			}
		}
		if isIncomplete {
			p.print("// contains unexported fields");
		}

	} else { // interface

		for i, f := range list {
			p.leadComment(f.Doc);
			if ftyp, isFtyp := f.Type.(*ast.FuncType); isFtyp {
				// method
				p.expr(f.Names[0]);  // exactly one name
				p.signature(ftyp.Params, ftyp.Results);
			} else {
				// embedded interface
				p.expr(f.Type);
			}
			p.print(token.SEMICOLON);
			p.lineComment(f.Comment);
			if i+1 < len(list) || isIncomplete {
				p.print(newline);
			}
		}
		if isIncomplete {
			p.print("// contains unexported methods");
		}

	}
	p.print(unindent, formfeed, rbrace, token.RBRACE);
}


// ----------------------------------------------------------------------------
// Expressions

func needsBlanks(expr ast.Expr) bool {
	switch x := expr.(type) {
	case *ast.Ident:
		// "long" identifiers look better with blanks around them
		return len(x.Value) > 8;
	case *ast.BasicLit:
		// "long" literals look better with blanks around them
		return len(x.Value) > 8;
	case *ast.ParenExpr:
		// parenthesized expressions don't need blanks around them
		return false;
	case *ast.IndexExpr:
		// index expressions don't need blanks if the indexed expressions are simple
		return needsBlanks(x.X)
	case *ast.CallExpr:
		// call expressions need blanks if they have more than one
		// argument or if the function expression needs blanks
		return len(x.Args) > 1 || needsBlanks(x.Fun);
	}
	return true;
}


func (p *printer) binaryExpr(x *ast.BinaryExpr, prec1 int) {
	prec := x.Op.Precedence();
	if prec < prec1 {
		// parenthesis needed
		// Note: The parser inserts an ast.ParenExpr node; thus this case
		//       can only occur if the AST is created in a different way.
		p.print(token.LPAREN);
		p.expr(x);
		p.print(token.RPAREN);
		return;
	}

	// Traverse left, collect all operations at same precedence
	// and determine if blanks should be printed around operators.
	//
	// This algorithm assumes that the right-hand side of a binary
	// operation has a different (higher) precedence then the current
	// node, which is how the parser creates the AST.
	var list vector.Vector;
	line := x.Y.Pos().Line;
	printBlanks := prec <= token.EQL.Precedence() || needsBlanks(x.Y);
	for {
		list.Push(x);
		if t, ok := x.X.(*ast.BinaryExpr); ok && t.Op.Precedence() == prec {
			x = t;
			prev := line;
			line = x.Y.Pos().Line;
			if needsBlanks(x.Y) || prev != line {
				printBlanks = true;
			}
		} else {
			break;
		}
	}
	prev := line;
	line = x.X.Pos().Line;
	if needsBlanks(x.X) || prev != line {
		printBlanks = true;
	}

	// Print collected operations left-to-right, with blanks if necessary.
	indented := false;
	p.expr1(x.X, prec);
	for list.Len() > 0 {
		x = list.Pop().(*ast.BinaryExpr);
		prev := line;
		line = x.Y.Pos().Line;
		if printBlanks {
			if prev != line {
				p.print(blank, x.OpPos, x.Op);
				// at least one line break, but respect an extra empty line
				// in the source
				if p.linebreak(line, 1, 2, false) && !indented {
					p.print(htab, indent);  // indent applies to next line
					indented = true;
				}
			} else {
				p.print(blank, x.OpPos, x.Op, blank);
			}
		} else {
			if prev != line {
				panic("internal error");
			}
			p.print(x.OpPos, x.Op);
		}
		p.expr1(x.Y, prec);
	}
	if indented {
		p.print(unindent);
	}
}


// Returns true if a separating semicolon is optional.
func (p *printer) expr1(expr ast.Expr, prec1 int) (optSemi bool) {
	p.print(expr.Pos());

	switch x := expr.(type) {
	case *ast.BadExpr:
		p.print("BadExpr");

	case *ast.Ident:
		p.print(x.Value);

	case *ast.BinaryExpr:
		p.binaryExpr(x, prec1);

	case *ast.KeyValueExpr:
		p.expr(x.Key);
		p.print(x.Colon, token.COLON, blank);
		p.expr(x.Value);

	case *ast.StarExpr:
		p.print(token.MUL);
		optSemi = p.expr(x.X);

	case *ast.UnaryExpr:
		const prec = token.UnaryPrec;
		if prec < prec1 {
			// parenthesis needed
			p.print(token.LPAREN);
			p.expr(x);
			p.print(token.RPAREN);
		} else {
			// no parenthesis needed
			p.print(x.Op);
			if x.Op == token.RANGE {
				p.print(blank);
			}
			p.expr1(x.X, prec);
		}

	case *ast.BasicLit:
		if p.mode & RawFormat == 0 {
			// tabwriter is used: escape all literals
			// so they pass through unchanged
			// (note that a legal Go program cannot contain an '\xff' byte in
			// literal source text since '\xff' is not a legal byte in correct
			// UTF-8 encoded text)
			p.print(esc, x.Value, esc);
		} else {
			p.print(x.Value);
		}

	case *ast.StringList:
		p.stringList(x.Strings);

	case *ast.FuncLit:
		p.expr(x.Type);
		p.print(blank);
		p.block(x.Body, 1);

	case *ast.ParenExpr:
		p.print(token.LPAREN);
		p.expr(x.X);
		p.print(x.Rparen, token.RPAREN);

	case *ast.SelectorExpr:
		p.expr1(x.X, token.HighestPrec);
		p.print(token.PERIOD);
		p.expr1(x.Sel, token.HighestPrec);

	case *ast.TypeAssertExpr:
		p.expr1(x.X, token.HighestPrec);
		p.print(token.PERIOD, token.LPAREN);
		if x.Type != nil {
			p.expr(x.Type);
		} else {
			p.print(token.TYPE);
		}
		p.print(token.RPAREN);

	case *ast.IndexExpr:
		p.expr1(x.X, token.HighestPrec);
		p.print(token.LBRACK);
		p.expr1(x.Index, token.LowestPrec);
		if x.End != nil {
			if needsBlanks(x.Index) || needsBlanks(x.End) {
				// blanks around ":"
				p.print(blank, token.COLON, blank);
			} else {
				// no blanks around ":"
				p.print(token.COLON);
			}
			p.expr(x.End);
		}
		p.print(token.RBRACK);

	case *ast.CallExpr:
		p.expr1(x.Fun, token.HighestPrec);
		p.print(x.Lparen, token.LPAREN);
		p.exprList(x.Args, commaSep);
		p.print(x.Rparen, token.RPAREN);

	case *ast.CompositeLit:
		p.expr1(x.Type, token.HighestPrec);
		p.print(x.Lbrace, token.LBRACE);
		p.exprList(x.Elts, commaSep | commaTerm);
		p.print(x.Rbrace, token.RBRACE);

	case *ast.Ellipsis:
		p.print(token.ELLIPSIS);

	case *ast.ArrayType:
		p.print(token.LBRACK);
		if x.Len != nil {
			p.expr(x.Len);
		}
		p.print(token.RBRACK);
		optSemi = p.expr(x.Elt);

	case *ast.StructType:
		p.print(token.STRUCT);
		p.fieldList(x.Lbrace, x.Fields, x.Rbrace, x.Incomplete, true);
		optSemi = true;

	case *ast.FuncType:
		p.print(token.FUNC);
		optSemi = p.signature(x.Params, x.Results);

	case *ast.InterfaceType:
		p.print(token.INTERFACE);
		p.fieldList(x.Lbrace, x.Methods, x.Rbrace, x.Incomplete, false);
		optSemi = true;

	case *ast.MapType:
		p.print(token.MAP, token.LBRACK);
		p.expr(x.Key);
		p.print(token.RBRACK);
		optSemi = p.expr(x.Value);

	case *ast.ChanType:
		switch x.Dir {
		case ast.SEND | ast.RECV:
			p.print(token.CHAN);
		case ast.RECV:
			p.print(token.ARROW, token.CHAN);
		case ast.SEND:
			p.print(token.CHAN, token.ARROW);
		}
		p.print(blank);
		optSemi = p.expr(x.Value);

	default:
		panic("unreachable");
	}

	return;
}


// Returns true if a separating semicolon is optional.
func (p *printer) expr(x ast.Expr) (optSemi bool) {
	return p.expr1(x, token.LowestPrec);
}


// ----------------------------------------------------------------------------
// Statements

const maxStmtNewlines = 2  // maximum number of newlines between statements

// Print the statement list indented, but without a newline after the last statement.
// Extra line breaks between statements in the source are respected but at most one
// empty line is printed between statements.
func (p *printer) stmtList(list []ast.Stmt, _indent int) {
	// TODO(gri): fix _indent code
	if _indent > 0 {
		p.print(indent);
	}
	for i, s := range list {
		// _indent == 0 only for lists of switch/select case clauses;
		// in those cases each clause is a new section
		p.linebreak(s.Pos().Line, 1, maxStmtNewlines, i == 0 || _indent == 0);
		if !p.stmt(s) {
			p.print(token.SEMICOLON);
		}
	}
	if _indent > 0 {
		p.print(unindent);
	}
}


func (p *printer) block(s *ast.BlockStmt, indent int) {
	p.print(s.Pos(), token.LBRACE);
	if len(s.List) > 0 || p.commentBefore(s.Rbrace) {
		p.stmtList(s.List, indent);
		p.linebreak(s.Rbrace.Line, 1, maxStmtNewlines, true);
	}
	p.print(s.Rbrace, token.RBRACE);
}


// TODO(gri): Decide if this should be used more broadly. The printing code
//            knows when to insert parentheses for precedence reasons, but
//            need to be careful to keep them around type expressions.
func stripParens(x ast.Expr) ast.Expr {
	if px, hasParens := x.(*ast.ParenExpr); hasParens {
		return stripParens(px.X);
	}
	return x;
}


func (p *printer) controlClause(isForStmt bool, init ast.Stmt, expr ast.Expr, post ast.Stmt) {
	p.print(blank);
	needsBlank := false;
	if init == nil && post == nil {
		// no semicolons required
		if expr != nil {
			p.expr(stripParens(expr));
			needsBlank = true;
		}
	} else {
		// all semicolons required
		// (they are not separators, print them explicitly)
		if init != nil {
			p.stmt(init);
		}
		p.print(token.SEMICOLON, blank);
		if expr != nil {
			p.expr(stripParens(expr));
			needsBlank = true;
		}
		if isForStmt {
			p.print(token.SEMICOLON, blank);
			needsBlank = false;
			if post != nil {
				p.stmt(post);
				needsBlank = true;
			}
		}
	}
	if needsBlank {
		p.print(blank);
	}
}


// Returns true if a separating semicolon is optional.
func (p *printer) stmt(stmt ast.Stmt) (optSemi bool) {
	p.print(stmt.Pos());

	switch s := stmt.(type) {
	case *ast.BadStmt:
		p.print("BadStmt");

	case *ast.DeclStmt:
		p.decl(s.Decl, inStmtList);
		optSemi = true;  // decl prints terminating semicolon if necessary

	case *ast.EmptyStmt:
		// nothing to do

	case *ast.LabeledStmt:
		// a "correcting" unindent immediately following a line break
		// is applied before the line break  if there is no comment
		// between (see writeWhitespace)
		p.print(unindent);
		p.expr(s.Label);
		p.print(token.COLON, vtab, indent);
		p.linebreak(s.Stmt.Pos().Line, 0, 1, true);
		optSemi = p.stmt(s.Stmt);

	case *ast.ExprStmt:
		p.expr(s.X);

	case *ast.IncDecStmt:
		p.expr(s.X);
		p.print(s.Tok);

	case *ast.AssignStmt:
		p.exprList(s.Lhs, commaSep);
		p.print(blank, s.TokPos, s.Tok);
		p.exprList(s.Rhs, blankStart | commaSep);

	case *ast.GoStmt:
		p.print(token.GO, blank);
		p.expr(s.Call);

	case *ast.DeferStmt:
		p.print(token.DEFER, blank);
		p.expr(s.Call);

	case *ast.ReturnStmt:
		p.print(token.RETURN);
		if s.Results != nil {
			p.exprList(s.Results, blankStart | commaSep);
		}

	case *ast.BranchStmt:
		p.print(s.Tok);
		if s.Label != nil {
			p.print(blank);
			p.expr(s.Label);
		}

	case *ast.BlockStmt:
		p.block(s, 1);
		optSemi = true;

	case *ast.IfStmt:
		p.print(token.IF);
		p.controlClause(false, s.Init, s.Cond, nil);
		p.block(s.Body, 1);
		optSemi = true;
		if s.Else != nil {
			p.print(blank, token.ELSE, blank);
			switch s.Else.(type) {
			case *ast.BlockStmt, *ast.IfStmt:
				optSemi = p.stmt(s.Else);
			default:
				p.print(token.LBRACE, indent, formfeed);
				p.stmt(s.Else);
				p.print(unindent, formfeed, token.RBRACE);
			}
		}

	case *ast.CaseClause:
		if s.Values != nil {
			p.print(token.CASE);
			p.exprList(s.Values, blankStart | commaSep);
		} else {
			p.print(token.DEFAULT);
		}
		p.print(s.Colon, token.COLON);
		p.stmtList(s.Body, 1);
		optSemi = true;  // "block" without {}'s

	case *ast.SwitchStmt:
		p.print(token.SWITCH);
		p.controlClause(false, s.Init, s.Tag, nil);
		p.block(s.Body, 0);
		optSemi = true;

	case *ast.TypeCaseClause:
		if s.Types != nil {
			p.print(token.CASE);
			p.exprList(s.Types, blankStart | commaSep);
		} else {
			p.print(token.DEFAULT);
		}
		p.print(s.Colon, token.COLON);
		p.stmtList(s.Body, 1);
		optSemi = true;  // "block" without {}'s

	case *ast.TypeSwitchStmt:
		p.print(token.SWITCH);
		if s.Init != nil {
			p.print(blank);
			p.stmt(s.Init);
			p.print(token.SEMICOLON);
		}
		p.print(blank);
		p.stmt(s.Assign);
		p.print(blank);
		p.block(s.Body, 0);
		optSemi = true;

	case *ast.CommClause:
		if s.Rhs != nil {
			p.print(token.CASE, blank);
			if s.Lhs != nil {
				p.expr(s.Lhs);
				p.print(blank, s.Tok, blank);
			}
			p.expr(s.Rhs);
		} else {
			p.print(token.DEFAULT);
		}
		p.print(s.Colon, token.COLON);
		p.stmtList(s.Body, 1);
		optSemi = true;  // "block" without {}'s

	case *ast.SelectStmt:
		p.print(token.SELECT, blank);
		p.block(s.Body, 0);
		optSemi = true;

	case *ast.ForStmt:
		p.print(token.FOR);
		p.controlClause(true, s.Init, s.Cond, s.Post);
		p.block(s.Body, 1);
		optSemi = true;

	case *ast.RangeStmt:
		p.print(token.FOR, blank);
		p.expr(s.Key);
		if s.Value != nil {
			p.print(token.COMMA, blank);
			p.expr(s.Value);
		}
		p.print(blank, s.TokPos, s.Tok, blank, token.RANGE, blank);
		p.expr(s.X);
		p.print(blank);
		p.block(s.Body, 1);
		optSemi = true;

	default:
		panic("unreachable");
	}

	return;
}


// ----------------------------------------------------------------------------
// Declarations

type declContext uint;
const (
	atTop declContext = iota;
	inGroup;
	inStmtList;
)

// The parameter n is the number of specs in the group; context specifies
// the surroundings of the declaration. Separating semicolons are printed
// depending on the context.
//
func (p *printer) spec(spec ast.Spec, n int, context declContext) {
	var (
		optSemi bool;  // true if a semicolon is optional
		comment *ast.CommentGroup;  // a line comment, if any
		extraTabs int;  // number of extra tabs before comment, if any
	)

	switch s := spec.(type) {
	case *ast.ImportSpec:
		p.leadComment(s.Doc);
		if s.Name != nil {
			p.expr(s.Name);
			p.print(blank);
		}
		p.expr(&ast.StringList{s.Path});
		comment = s.Comment;

	case *ast.ValueSpec:
		p.leadComment(s.Doc);
		p.identList(s.Names);  // always present
		if n == 1 {
			if s.Type != nil {
				p.print(blank);
				optSemi = p.expr(s.Type);
			}
			if s.Values != nil {
				p.print(blank, token.ASSIGN);
				p.exprList(s.Values, blankStart | commaSep);
				optSemi = false;
			}
		} else {
			extraTabs = 2;
			if s.Type != nil || s.Values != nil {
				p.print(vtab);
			}
			if s.Type != nil {
				optSemi = p.expr(s.Type);
				extraTabs = 1;
			}
			if s.Values != nil {
				p.print(vtab);
				p.print(token.ASSIGN);
				p.exprList(s.Values, blankStart | commaSep);
				optSemi = false;
				extraTabs = 0;
			}
		}
		comment = s.Comment;

	case *ast.TypeSpec:
		p.leadComment(s.Doc);
		p.expr(s.Name);
		if n == 1 {
			p.print(blank);
		} else {
			p.print(vtab);
		}
		optSemi = p.expr(s.Type);
		comment = s.Comment;

	default:
		panic("unreachable");
	}

	if context == inGroup || context == inStmtList && !optSemi {
		p.print(token.SEMICOLON);
	}

	if comment != nil {
		for ; extraTabs > 0; extraTabs-- {
			p.print(vtab);
		}
		p.lineComment(comment);
	}
}


func (p *printer) decl(decl ast.Decl, context declContext) {
	switch d := decl.(type) {
	case *ast.BadDecl:
		p.print(d.Pos(), "BadDecl");

	case *ast.GenDecl:
		p.leadComment(d.Doc);
		p.print(lineTag(d.Pos()), d.Tok, blank);

		if d.Lparen.IsValid() {
			// group of parenthesized declarations
			p.print(d.Lparen, token.LPAREN);
			if len(d.Specs) > 0 {
				p.print(indent, formfeed);
				for i, s := range d.Specs {
					if i > 0 {
						p.print(newline);
					}
					p.spec(s, len(d.Specs), inGroup);
				}
				p.print(unindent, formfeed);
			}
			p.print(d.Rparen, token.RPAREN);

		} else {
			// single declaration
			p.spec(d.Specs[0], 1, context);
		}

	case *ast.FuncDecl:
		p.leadComment(d.Doc);
		p.print(lineTag(d.Pos()), token.FUNC, blank);
		if recv := d.Recv; recv != nil {
			// method: print receiver
			p.print(token.LPAREN);
			if len(recv.Names) > 0 {
				p.expr(recv.Names[0]);
				p.print(blank);
			}
			p.expr(recv.Type);
			p.print(token.RPAREN, blank);
		}
		p.expr(d.Name);
		p.signature(d.Type.Params, d.Type.Results);
		if d.Body != nil {
			p.print(blank);
			p.block(d.Body, 1);
		}

	default:
		panic("unreachable");
	}
}


// ----------------------------------------------------------------------------
// Files

const maxDeclNewlines = 3  // maximum number of newlines between declarations

func declToken(decl ast.Decl) (tok token.Token) {
	tok = token.ILLEGAL;
	switch d := decl.(type) {
	case *ast.GenDecl:
		tok = d.Tok;
	case *ast.FuncDecl:
		tok = token.FUNC;
	}
	return;
}


func (p *printer) file(src *ast.File) {
	p.leadComment(src.Doc);
	p.print(src.Pos(), token.PACKAGE, blank);
	p.expr(src.Name);

	if len(src.Decls) > 0 {
		tok := token.ILLEGAL;
		for _, d := range src.Decls {
			prev := tok;
			tok = declToken(d);
			// if the declaration token changed (e.g., from CONST to TYPE)
			// print an empty line between top-level declarations
			min := 1;
			if prev != tok {
				min = 2;
			}
			p.linebreak(d.Pos().Line, min, maxDeclNewlines, false);
			p.decl(d, atTop);
		}
	}

	p.print(newline);
}


// ----------------------------------------------------------------------------
// Trimmer

// A trimmer is an io.Writer filter for stripping trailing blanks
// and tabs, and for converting formfeed and vtab characters into
// newlines and htabs (in case no tabwriter is used).
//
type trimmer struct {
	output io.Writer;
	buf bytes.Buffer;
}


// Design note: It is tempting to eliminate extra blanks occuring in
//              whitespace in this function as it could simplify some
//              of the blanks logic in the node printing functions.
//              However, this would mess up any formatting done by
//              the tabwriter.

func (p *trimmer) Write(data []byte) (n int, err os.Error) {
	// m < 0: no unwritten data except for whitespace
	// m >= 0: data[m:n] unwritten and no whitespace
	m := 0;
	if p.buf.Len() > 0 {
		m = -1;
	}

	var b byte;
	for n, b = range data {
		switch b {
		default:
			// write any pending whitespace
			if m < 0 {
				if _, err = p.output.Write(p.buf.Bytes()); err != nil {
					return;
				}
				p.buf.Reset();
				m = n;
			}

		case '\v':
			b = '\t';  // convert to htab
			fallthrough;

		case '\t', ' ':
			// write any pending (non-whitespace) data
			if m >= 0 {
				if _, err = p.output.Write(data[m:n]); err != nil {
					return;
				}
				m = -1;
			}
			// collect whitespace
			p.buf.WriteByte(b);  // WriteByte returns no errors

		case '\f', '\n':
			// discard whitespace
			p.buf.Reset();
			// write any pending (non-whitespace) data
			if m >= 0 {
				if _, err = p.output.Write(data[m:n]); err != nil {
					return;
				}
				m = -1;
			}
			// convert formfeed into newline
			if _, err = p.output.Write(newlines[0:1]); err != nil {
				return;
			}
		}
	}
	n = len(data);

	// write any pending non-whitespace
	if m >= 0 {
		if _, err = p.output.Write(data[m:n]); err != nil {
			return;
		}
	}

	return;
}


// ----------------------------------------------------------------------------
// Public interface

var inf = token.Position{Offset: 1<<30, Line: 1<<30}


// Fprint "pretty-prints" an AST node to output and returns the number of
// bytes written, and an error, if any. The node type must be *ast.File,
// or assignment-compatible to ast.Expr, ast.Decl, or ast.Stmt. Printing
// is controlled by the mode and tabwidth parameters.
//
func Fprint(output io.Writer, node interface{}, mode uint, tabwidth int) (int, os.Error) {
	// redirect output through a trimmer to eliminate trailing whitespace
	// (Input to a tabwriter must be untrimmed since trailing tabs provide
	// formatting information. The tabwriter could provide trimming
	// functionality but no tabwriter is used when RawFormat is set.)
	output = &trimmer{output: output};

	// setup tabwriter if needed and redirect output
	var tw *tabwriter.Writer;
	if mode & RawFormat == 0 {
		padchar := byte('\t');
		if mode & UseSpaces != 0 {
			padchar = ' ';
		}
		twmode := tabwriter.DiscardEmptyColumns;
		if mode & GenHTML != 0 {
			twmode |= tabwriter.FilterHTML;
		}
		tw = tabwriter.NewWriter(output, tabwidth, 1, padchar, twmode);
		output = tw;
	}

	// setup printer and print node
	var p printer;
	p.init(output, mode);
	go func() {
		switch n := node.(type) {
		case ast.Expr:
			p.expr(n);
		case ast.Stmt:
			p.stmt(n);
		case ast.Decl:
			p.decl(n, atTop);
		case *ast.File:
			p.comment = n.Comments;
			p.file(n);
		default:
			p.errors <- os.NewError(fmt.Sprintf("unsupported node type %T", n));
			runtime.Goexit();
		}
		p.flush(inf);
		p.errors <- nil;  // no errors
	}();
	err := <-p.errors;  // wait for completion of goroutine

	// flush tabwriter, if any
	if tw != nil {
		tw.Flush();  // ignore errors
	}

	return p.written, err;
}
