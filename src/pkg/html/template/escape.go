// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes"
	"fmt"
	"html"
	"io"
	"text/template"
	"text/template/parse"
)

// escapeTemplates rewrites the named templates, which must be
// associated with t, to guarantee that the output of any of the named
// templates is properly escaped.  Names should include the names of
// all templates that might be Executed but need not include helper
// templates.  If no error is returned, then the named templates have
// been modified.  Otherwise the named templates have been rendered
// unusable.
func escapeTemplates(tmpl *Template, names ...string) error {
	e := newEscaper(tmpl)
	for _, name := range names {
		c, _ := e.escapeTree(context{}, name, 0)
		var err error
		if c.err != nil {
			err, c.err.Name = c.err, name
		} else if c.state != stateText {
			err = &Error{ErrEndContext, name, 0, fmt.Sprintf("ends in a non-text context: %v", c)}
		}
		if err != nil {
			// Prevent execution of unsafe templates.
			for _, name := range names {
				if t := tmpl.set[name]; t != nil {
					t.text.Tree = nil
					t.Tree = nil
				}
			}
			return err
		}
		tmpl.escaped = true
		tmpl.Tree = tmpl.text.Tree
	}
	e.commit()
	return nil
}

// funcMap maps command names to functions that render their inputs safe.
var funcMap = template.FuncMap{
	"html_template_attrescaper":     attrEscaper,
	"html_template_commentescaper":  commentEscaper,
	"html_template_cssescaper":      cssEscaper,
	"html_template_cssvaluefilter":  cssValueFilter,
	"html_template_htmlnamefilter":  htmlNameFilter,
	"html_template_htmlescaper":     htmlEscaper,
	"html_template_jsregexpescaper": jsRegexpEscaper,
	"html_template_jsstrescaper":    jsStrEscaper,
	"html_template_jsvalescaper":    jsValEscaper,
	"html_template_nospaceescaper":  htmlNospaceEscaper,
	"html_template_rcdataescaper":   rcdataEscaper,
	"html_template_urlescaper":      urlEscaper,
	"html_template_urlfilter":       urlFilter,
	"html_template_urlnormalizer":   urlNormalizer,
}

// equivEscapers matches contextual escapers to equivalent template builtins.
var equivEscapers = map[string]string{
	"html_template_attrescaper":    "html",
	"html_template_htmlescaper":    "html",
	"html_template_nospaceescaper": "html",
	"html_template_rcdataescaper":  "html",
	"html_template_urlescaper":     "urlquery",
	"html_template_urlnormalizer":  "urlquery",
}

// escaper collects type inferences about templates and changes needed to make
// templates injection safe.
type escaper struct {
	tmpl *Template
	// output[templateName] is the output context for a templateName that
	// has been mangled to include its input context.
	output map[string]context
	// derived[c.mangle(name)] maps to a template derived from the template
	// named name templateName for the start context c.
	derived map[string]*template.Template
	// called[templateName] is a set of called mangled template names.
	called map[string]bool
	// xxxNodeEdits are the accumulated edits to apply during commit.
	// Such edits are not applied immediately in case a template set
	// executes a given template in different escaping contexts.
	actionNodeEdits   map[*parse.ActionNode][]string
	templateNodeEdits map[*parse.TemplateNode]string
	textNodeEdits     map[*parse.TextNode][]byte
}

// newEscaper creates a blank escaper for the given set.
func newEscaper(t *Template) *escaper {
	return &escaper{
		t,
		map[string]context{},
		map[string]*template.Template{},
		map[string]bool{},
		map[*parse.ActionNode][]string{},
		map[*parse.TemplateNode]string{},
		map[*parse.TextNode][]byte{},
	}
}

// filterFailsafe is an innocuous word that is emitted in place of unsafe values
// by sanitizer functions. It is not a keyword in any programming language,
// contains no special characters, is not empty, and when it appears in output
// it is distinct enough that a developer can find the source of the problem
// via a search engine.
const filterFailsafe = "ZgotmplZ"

// escape escapes a template node.
func (e *escaper) escape(c context, n parse.Node) context {
	switch n := n.(type) {
	case *parse.ActionNode:
		return e.escapeAction(c, n)
	case *parse.IfNode:
		return e.escapeBranch(c, &n.BranchNode, "if")
	case *parse.ListNode:
		return e.escapeList(c, n)
	case *parse.RangeNode:
		return e.escapeBranch(c, &n.BranchNode, "range")
	case *parse.TemplateNode:
		return e.escapeTemplate(c, n)
	case *parse.TextNode:
		return e.escapeText(c, n)
	case *parse.WithNode:
		return e.escapeBranch(c, &n.BranchNode, "with")
	}
	panic("escaping " + n.String() + " is unimplemented")
}

// escapeAction escapes an action template node.
func (e *escaper) escapeAction(c context, n *parse.ActionNode) context {
	if len(n.Pipe.Decl) != 0 {
		// A local variable assignment, not an interpolation.
		return c
	}
	c = nudge(c)
	s := make([]string, 0, 3)
	switch c.state {
	case stateError:
		return c
	case stateURL, stateCSSDqStr, stateCSSSqStr, stateCSSDqURL, stateCSSSqURL, stateCSSURL:
		switch c.urlPart {
		case urlPartNone:
			s = append(s, "html_template_urlfilter")
			fallthrough
		case urlPartPreQuery:
			switch c.state {
			case stateCSSDqStr, stateCSSSqStr:
				s = append(s, "html_template_cssescaper")
			default:
				s = append(s, "html_template_urlnormalizer")
			}
		case urlPartQueryOrFrag:
			s = append(s, "html_template_urlescaper")
		case urlPartUnknown:
			return context{
				state: stateError,
				err:   errorf(ErrAmbigContext, n.Line, "%s appears in an ambiguous URL context", n),
			}
		default:
			panic(c.urlPart.String())
		}
	case stateJS:
		s = append(s, "html_template_jsvalescaper")
		// A slash after a value starts a div operator.
		c.jsCtx = jsCtxDivOp
	case stateJSDqStr, stateJSSqStr:
		s = append(s, "html_template_jsstrescaper")
	case stateJSRegexp:
		s = append(s, "html_template_jsregexpescaper")
	case stateCSS:
		s = append(s, "html_template_cssvaluefilter")
	case stateText:
		s = append(s, "html_template_htmlescaper")
	case stateRCDATA:
		s = append(s, "html_template_rcdataescaper")
	case stateAttr:
		// Handled below in delim check.
	case stateAttrName, stateTag:
		c.state = stateAttrName
		s = append(s, "html_template_htmlnamefilter")
	default:
		if isComment(c.state) {
			s = append(s, "html_template_commentescaper")
		} else {
			panic("unexpected state " + c.state.String())
		}
	}
	switch c.delim {
	case delimNone:
		// No extra-escaping needed for raw text content.
	case delimSpaceOrTagEnd:
		s = append(s, "html_template_nospaceescaper")
	default:
		s = append(s, "html_template_attrescaper")
	}
	e.editActionNode(n, s)
	return c
}

// ensurePipelineContains ensures that the pipeline has commands with
// the identifiers in s in order.
// If the pipeline already has some of the sanitizers, do not interfere.
// For example, if p is (.X | html) and s is ["escapeJSVal", "html"] then it
// has one matching, "html", and one to insert, "escapeJSVal", to produce
// (.X | escapeJSVal | html).
func ensurePipelineContains(p *parse.PipeNode, s []string) {
	if len(s) == 0 {
		return
	}
	n := len(p.Cmds)
	// Find the identifiers at the end of the command chain.
	idents := p.Cmds
	for i := n - 1; i >= 0; i-- {
		if cmd := p.Cmds[i]; len(cmd.Args) != 0 {
			if _, ok := cmd.Args[0].(*parse.IdentifierNode); ok {
				continue
			}
		}
		idents = p.Cmds[i+1:]
	}
	dups := 0
	for _, id := range idents {
		if escFnsEq(s[dups], (id.Args[0].(*parse.IdentifierNode)).Ident) {
			dups++
			if dups == len(s) {
				return
			}
		}
	}
	newCmds := make([]*parse.CommandNode, n-len(idents), n+len(s)-dups)
	copy(newCmds, p.Cmds)
	// Merge existing identifier commands with the sanitizers needed.
	for _, id := range idents {
		pos := id.Args[0].Position()
		i := indexOfStr((id.Args[0].(*parse.IdentifierNode)).Ident, s, escFnsEq)
		if i != -1 {
			for _, name := range s[:i] {
				newCmds = appendCmd(newCmds, newIdentCmd(name, pos))
			}
			s = s[i+1:]
		}
		newCmds = appendCmd(newCmds, id)
	}
	// Create any remaining sanitizers.
	for _, name := range s {
		newCmds = appendCmd(newCmds, newIdentCmd(name, p.Position()))
	}
	p.Cmds = newCmds
}

// redundantFuncs[a][b] implies that funcMap[b](funcMap[a](x)) == funcMap[a](x)
// for all x.
var redundantFuncs = map[string]map[string]bool{
	"html_template_commentescaper": {
		"html_template_attrescaper":    true,
		"html_template_nospaceescaper": true,
		"html_template_htmlescaper":    true,
	},
	"html_template_cssescaper": {
		"html_template_attrescaper": true,
	},
	"html_template_jsregexpescaper": {
		"html_template_attrescaper": true,
	},
	"html_template_jsstrescaper": {
		"html_template_attrescaper": true,
	},
	"html_template_urlescaper": {
		"html_template_urlnormalizer": true,
	},
}

// appendCmd appends the given command to the end of the command pipeline
// unless it is redundant with the last command.
func appendCmd(cmds []*parse.CommandNode, cmd *parse.CommandNode) []*parse.CommandNode {
	if n := len(cmds); n != 0 {
		last, ok := cmds[n-1].Args[0].(*parse.IdentifierNode)
		next, _ := cmd.Args[0].(*parse.IdentifierNode)
		if ok && redundantFuncs[last.Ident][next.Ident] {
			return cmds
		}
	}
	return append(cmds, cmd)
}

// indexOfStr is the first i such that eq(s, strs[i]) or -1 if s was not found.
func indexOfStr(s string, strs []string, eq func(a, b string) bool) int {
	for i, t := range strs {
		if eq(s, t) {
			return i
		}
	}
	return -1
}

// escFnsEq reports whether the two escaping functions are equivalent.
func escFnsEq(a, b string) bool {
	if e := equivEscapers[a]; e != "" {
		a = e
	}
	if e := equivEscapers[b]; e != "" {
		b = e
	}
	return a == b
}

// newIdentCmd produces a command containing a single identifier node.
func newIdentCmd(identifier string, pos parse.Pos) *parse.CommandNode {
	return &parse.CommandNode{
		NodeType: parse.NodeCommand,
		Args:     []parse.Node{parse.NewIdentifier(identifier).SetPos(pos)},
	}
}

// nudge returns the context that would result from following empty string
// transitions from the input context.
// For example, parsing:
//     `<a href=`
// will end in context{stateBeforeValue, attrURL}, but parsing one extra rune:
//     `<a href=x`
// will end in context{stateURL, delimSpaceOrTagEnd, ...}.
// There are two transitions that happen when the 'x' is seen:
// (1) Transition from a before-value state to a start-of-value state without
//     consuming any character.
// (2) Consume 'x' and transition past the first value character.
// In this case, nudging produces the context after (1) happens.
func nudge(c context) context {
	switch c.state {
	case stateTag:
		// In `<foo {{.}}`, the action should emit an attribute.
		c.state = stateAttrName
	case stateBeforeValue:
		// In `<foo bar={{.}}`, the action is an undelimited value.
		c.state, c.delim, c.attr = attrStartStates[c.attr], delimSpaceOrTagEnd, attrNone
	case stateAfterName:
		// In `<foo bar {{.}}`, the action is an attribute name.
		c.state, c.attr = stateAttrName, attrNone
	}
	return c
}

// join joins the two contexts of a branch template node. The result is an
// error context if either of the input contexts are error contexts, or if the
// the input contexts differ.
func join(a, b context, line int, nodeName string) context {
	if a.state == stateError {
		return a
	}
	if b.state == stateError {
		return b
	}
	if a.eq(b) {
		return a
	}

	c := a
	c.urlPart = b.urlPart
	if c.eq(b) {
		// The contexts differ only by urlPart.
		c.urlPart = urlPartUnknown
		return c
	}

	c = a
	c.jsCtx = b.jsCtx
	if c.eq(b) {
		// The contexts differ only by jsCtx.
		c.jsCtx = jsCtxUnknown
		return c
	}

	// Allow a nudged context to join with an unnudged one.
	// This means that
	//   <p title={{if .C}}{{.}}{{end}}
	// ends in an unquoted value state even though the else branch
	// ends in stateBeforeValue.
	if c, d := nudge(a), nudge(b); !(c.eq(a) && d.eq(b)) {
		if e := join(c, d, line, nodeName); e.state != stateError {
			return e
		}
	}

	return context{
		state: stateError,
		err:   errorf(ErrBranchEnd, line, "{{%s}} branches end in different contexts: %v, %v", nodeName, a, b),
	}
}

// escapeBranch escapes a branch template node: "if", "range" and "with".
func (e *escaper) escapeBranch(c context, n *parse.BranchNode, nodeName string) context {
	c0 := e.escapeList(c, n.List)
	if nodeName == "range" && c0.state != stateError {
		// The "true" branch of a "range" node can execute multiple times.
		// We check that executing n.List once results in the same context
		// as executing n.List twice.
		c1, _ := e.escapeListConditionally(c0, n.List, nil)
		c0 = join(c0, c1, n.Line, nodeName)
		if c0.state == stateError {
			// Make clear that this is a problem on loop re-entry
			// since developers tend to overlook that branch when
			// debugging templates.
			c0.err.Line = n.Line
			c0.err.Description = "on range loop re-entry: " + c0.err.Description
			return c0
		}
	}
	c1 := e.escapeList(c, n.ElseList)
	return join(c0, c1, n.Line, nodeName)
}

// escapeList escapes a list template node.
func (e *escaper) escapeList(c context, n *parse.ListNode) context {
	if n == nil {
		return c
	}
	for _, m := range n.Nodes {
		c = e.escape(c, m)
	}
	return c
}

// escapeListConditionally escapes a list node but only preserves edits and
// inferences in e if the inferences and output context satisfy filter.
// It returns the best guess at an output context, and the result of the filter
// which is the same as whether e was updated.
func (e *escaper) escapeListConditionally(c context, n *parse.ListNode, filter func(*escaper, context) bool) (context, bool) {
	e1 := newEscaper(e.tmpl)
	// Make type inferences available to f.
	for k, v := range e.output {
		e1.output[k] = v
	}
	c = e1.escapeList(c, n)
	ok := filter != nil && filter(e1, c)
	if ok {
		// Copy inferences and edits from e1 back into e.
		for k, v := range e1.output {
			e.output[k] = v
		}
		for k, v := range e1.derived {
			e.derived[k] = v
		}
		for k, v := range e1.called {
			e.called[k] = v
		}
		for k, v := range e1.actionNodeEdits {
			e.editActionNode(k, v)
		}
		for k, v := range e1.templateNodeEdits {
			e.editTemplateNode(k, v)
		}
		for k, v := range e1.textNodeEdits {
			e.editTextNode(k, v)
		}
	}
	return c, ok
}

// escapeTemplate escapes a {{template}} call node.
func (e *escaper) escapeTemplate(c context, n *parse.TemplateNode) context {
	c, name := e.escapeTree(c, n.Name, n.Line)
	if name != n.Name {
		e.editTemplateNode(n, name)
	}
	return c
}

// escapeTree escapes the named template starting in the given context as
// necessary and returns its output context.
func (e *escaper) escapeTree(c context, name string, line int) (context, string) {
	// Mangle the template name with the input context to produce a reliable
	// identifier.
	dname := c.mangle(name)
	e.called[dname] = true
	if out, ok := e.output[dname]; ok {
		// Already escaped.
		return out, dname
	}
	t := e.template(name)
	if t == nil {
		// Two cases: The template exists but is empty, or has never been mentioned at
		// all. Distinguish the cases in the error messages.
		if e.tmpl.set[name] != nil {
			return context{
				state: stateError,
				err:   errorf(ErrNoSuchTemplate, line, "%q is an incomplete or empty template", name),
			}, dname
		}
		return context{
			state: stateError,
			err:   errorf(ErrNoSuchTemplate, line, "no such template %q", name),
		}, dname
	}
	if dname != name {
		// Use any template derived during an earlier call to escapeTemplate
		// with different top level templates, or clone if necessary.
		dt := e.template(dname)
		if dt == nil {
			dt = template.New(dname)
			dt.Tree = &parse.Tree{Name: dname, Root: t.Root.CopyList()}
			e.derived[dname] = dt
		}
		t = dt
	}
	return e.computeOutCtx(c, t), dname
}

// computeOutCtx takes a template and its start context and computes the output
// context while storing any inferences in e.
func (e *escaper) computeOutCtx(c context, t *template.Template) context {
	// Propagate context over the body.
	c1, ok := e.escapeTemplateBody(c, t)
	if !ok {
		// Look for a fixed point by assuming c1 as the output context.
		if c2, ok2 := e.escapeTemplateBody(c1, t); ok2 {
			c1, ok = c2, true
		}
		// Use c1 as the error context if neither assumption worked.
	}
	if !ok && c1.state != stateError {
		return context{
			state: stateError,
			// TODO: Find the first node with a line in t.text.Tree.Root
			err: errorf(ErrOutputContext, 0, "cannot compute output context for template %s", t.Name()),
		}
	}
	return c1
}

// escapeTemplateBody escapes the given template assuming the given output
// context, and returns the best guess at the output context and whether the
// assumption was correct.
func (e *escaper) escapeTemplateBody(c context, t *template.Template) (context, bool) {
	filter := func(e1 *escaper, c1 context) bool {
		if c1.state == stateError {
			// Do not update the input escaper, e.
			return false
		}
		if !e1.called[t.Name()] {
			// If t is not recursively called, then c1 is an
			// accurate output context.
			return true
		}
		// c1 is accurate if it matches our assumed output context.
		return c.eq(c1)
	}
	// We need to assume an output context so that recursive template calls
	// take the fast path out of escapeTree instead of infinitely recursing.
	// Naively assuming that the input context is the same as the output
	// works >90% of the time.
	e.output[t.Name()] = c
	return e.escapeListConditionally(c, t.Tree.Root, filter)
}

// delimEnds maps each delim to a string of characters that terminate it.
var delimEnds = [...]string{
	delimDoubleQuote: `"`,
	delimSingleQuote: "'",
	// Determined empirically by running the below in various browsers.
	// var div = document.createElement("DIV");
	// for (var i = 0; i < 0x10000; ++i) {
	//   div.innerHTML = "<span title=x" + String.fromCharCode(i) + "-bar>";
	//   if (div.getElementsByTagName("SPAN")[0].title.indexOf("bar") < 0)
	//     document.write("<p>U+" + i.toString(16));
	// }
	delimSpaceOrTagEnd: " \t\n\f\r>",
}

var doctypeBytes = []byte("<!DOCTYPE")

// escapeText escapes a text template node.
func (e *escaper) escapeText(c context, n *parse.TextNode) context {
	s, written, i, b := n.Text, 0, 0, new(bytes.Buffer)
	for i != len(s) {
		c1, nread := contextAfterText(c, s[i:])
		i1 := i + nread
		if c.state == stateText || c.state == stateRCDATA {
			end := i1
			if c1.state != c.state {
				for j := end - 1; j >= i; j-- {
					if s[j] == '<' {
						end = j
						break
					}
				}
			}
			for j := i; j < end; j++ {
				if s[j] == '<' && !bytes.HasPrefix(bytes.ToUpper(s[j:]), doctypeBytes) {
					b.Write(s[written:j])
					b.WriteString("&lt;")
					written = j + 1
				}
			}
		} else if isComment(c.state) && c.delim == delimNone {
			switch c.state {
			case stateJSBlockCmt:
				// http://es5.github.com/#x7.4:
				// "Comments behave like white space and are
				// discarded except that, if a MultiLineComment
				// contains a line terminator character, then
				// the entire comment is considered to be a
				// LineTerminator for purposes of parsing by
				// the syntactic grammar."
				if bytes.IndexAny(s[written:i1], "\n\r\u2028\u2029") != -1 {
					b.WriteByte('\n')
				} else {
					b.WriteByte(' ')
				}
			case stateCSSBlockCmt:
				b.WriteByte(' ')
			}
			written = i1
		}
		if c.state != c1.state && isComment(c1.state) && c1.delim == delimNone {
			// Preserve the portion between written and the comment start.
			cs := i1 - 2
			if c1.state == stateHTMLCmt {
				// "<!--" instead of "/*" or "//"
				cs -= 2
			}
			b.Write(s[written:cs])
			written = i1
		}
		if i == i1 && c.state == c1.state {
			panic(fmt.Sprintf("infinite loop from %v to %v on %q..%q", c, c1, s[:i], s[i:]))
		}
		c, i = c1, i1
	}

	if written != 0 && c.state != stateError {
		if !isComment(c.state) || c.delim != delimNone {
			b.Write(n.Text[written:])
		}
		e.editTextNode(n, b.Bytes())
	}
	return c
}

// contextAfterText starts in context c, consumes some tokens from the front of
// s, then returns the context after those tokens and the unprocessed suffix.
func contextAfterText(c context, s []byte) (context, int) {
	if c.delim == delimNone {
		c1, i := tSpecialTagEnd(c, s)
		if i == 0 {
			// A special end tag (`</script>`) has been seen and
			// all content preceding it has been consumed.
			return c1, 0
		}
		// Consider all content up to any end tag.
		return transitionFunc[c.state](c, s[:i])
	}

	i := bytes.IndexAny(s, delimEnds[c.delim])
	if i == -1 {
		i = len(s)
	}
	if c.delim == delimSpaceOrTagEnd {
		// http://www.w3.org/TR/html5/tokenization.html#attribute-value-unquoted-state
		// lists the runes below as error characters.
		// Error out because HTML parsers may differ on whether
		// "<a id= onclick=f("     ends inside id's or onclick's value,
		// "<a class=`foo "        ends inside a value,
		// "<a style=font:'Arial'" needs open-quote fixup.
		// IE treats '`' as a quotation character.
		if j := bytes.IndexAny(s[:i], "\"'<=`"); j >= 0 {
			return context{
				state: stateError,
				err:   errorf(ErrBadHTML, 0, "%q in unquoted attr: %q", s[j:j+1], s[:i]),
			}, len(s)
		}
	}
	if i == len(s) {
		// Remain inside the attribute.
		// Decode the value so non-HTML rules can easily handle
		//     <button onclick="alert(&quot;Hi!&quot;)">
		// without having to entity decode token boundaries.
		for u := []byte(html.UnescapeString(string(s))); len(u) != 0; {
			c1, i1 := transitionFunc[c.state](c, u)
			c, u = c1, u[i1:]
		}
		return c, len(s)
	}
	if c.delim != delimSpaceOrTagEnd {
		// Consume any quote.
		i++
	}
	// On exiting an attribute, we discard all state information
	// except the state and element.
	return context{state: stateTag, element: c.element}, i
}

// editActionNode records a change to an action pipeline for later commit.
func (e *escaper) editActionNode(n *parse.ActionNode, cmds []string) {
	if _, ok := e.actionNodeEdits[n]; ok {
		panic(fmt.Sprintf("node %s shared between templates", n))
	}
	e.actionNodeEdits[n] = cmds
}

// editTemplateNode records a change to a {{template}} callee for later commit.
func (e *escaper) editTemplateNode(n *parse.TemplateNode, callee string) {
	if _, ok := e.templateNodeEdits[n]; ok {
		panic(fmt.Sprintf("node %s shared between templates", n))
	}
	e.templateNodeEdits[n] = callee
}

// editTextNode records a change to a text node for later commit.
func (e *escaper) editTextNode(n *parse.TextNode, text []byte) {
	if _, ok := e.textNodeEdits[n]; ok {
		panic(fmt.Sprintf("node %s shared between templates", n))
	}
	e.textNodeEdits[n] = text
}

// commit applies changes to actions and template calls needed to contextually
// autoescape content and adds any derived templates to the set.
func (e *escaper) commit() {
	for name := range e.output {
		e.template(name).Funcs(funcMap)
	}
	for _, t := range e.derived {
		if _, err := e.tmpl.text.AddParseTree(t.Name(), t.Tree); err != nil {
			panic("error adding derived template")
		}
	}
	for n, s := range e.actionNodeEdits {
		ensurePipelineContains(n.Pipe, s)
	}
	for n, name := range e.templateNodeEdits {
		n.Name = name
	}
	for n, s := range e.textNodeEdits {
		n.Text = s
	}
}

// template returns the named template given a mangled template name.
func (e *escaper) template(name string) *template.Template {
	t := e.tmpl.text.Lookup(name)
	if t == nil {
		t = e.derived[name]
	}
	return t
}

// Forwarding functions so that clients need only import this package
// to reach the general escaping functions of text/template.

// HTMLEscape writes to w the escaped HTML equivalent of the plain text data b.
func HTMLEscape(w io.Writer, b []byte) {
	template.HTMLEscape(w, b)
}

// HTMLEscapeString returns the escaped HTML equivalent of the plain text data s.
func HTMLEscapeString(s string) string {
	return template.HTMLEscapeString(s)
}

// HTMLEscaper returns the escaped HTML equivalent of the textual
// representation of its arguments.
func HTMLEscaper(args ...interface{}) string {
	return template.HTMLEscaper(args...)
}

// JSEscape writes to w the escaped JavaScript equivalent of the plain text data b.
func JSEscape(w io.Writer, b []byte) {
	template.JSEscape(w, b)
}

// JSEscapeString returns the escaped JavaScript equivalent of the plain text data s.
func JSEscapeString(s string) string {
	return template.JSEscapeString(s)
}

// JSEscaper returns the escaped JavaScript equivalent of the textual
// representation of its arguments.
func JSEscaper(args ...interface{}) string {
	return template.JSEscaper(args...)
}

// URLQueryEscaper returns the escaped value of the textual representation of
// its arguments in a form suitable for embedding in a URL query.
func URLQueryEscaper(args ...interface{}) string {
	return template.URLQueryEscaper(args...)
}
