// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"bytes"
	"fmt"
	"html"
	"os"
	"template"
	"template/parse"
)

// Escape rewrites each action in the template to guarantee that the output is
// properly escaped.
func Escape(t *template.Template) (*template.Template, os.Error) {
	var s template.Set
	s.Add(t)
	if _, err := EscapeSet(&s, t.Name()); err != nil {
		return nil, err
	}
	// TODO: if s contains cloned dependencies due to self-recursion
	// cross-context, error out.
	return t, nil
}

// EscapeSet rewrites the template set to guarantee that the output of any of
// the named templates is properly escaped.
// Names should include the names of all templates that might be Executed but
// need not include helper templates.
// If no error is returned, then the named templates have been modified. 
// Otherwise the named templates have been rendered unusable.
func EscapeSet(s *template.Set, names ...string) (*template.Set, os.Error) {
	if len(names) == 0 {
		// TODO: Maybe add a method to Set to enumerate template names
		// and use those instead.
		return nil, os.NewError("must specify names of top level templates")
	}
	e := escaper{
		s,
		map[string]context{},
		map[string]*template.Template{},
		map[string]bool{},
		map[*parse.ActionNode][]string{},
		map[*parse.TemplateNode]string{},
	}
	for _, name := range names {
		c, _ := e.escapeTree(context{}, name, 0)
		var err os.Error
		if c.errStr != "" {
			err = fmt.Errorf("%s:%d: %s", name, c.errLine, c.errStr)
		} else if c.state != stateText {
			err = fmt.Errorf("%s ends in a non-text context: %v", name, c)
		}
		if err != nil {
			// Prevent execution of unsafe templates.
			for _, name := range names {
				if t := s.Template(name); t != nil {
					t.Tree = nil
				}
			}
			return nil, err
		}
	}
	e.commit()
	return s, nil
}

// funcMap maps command names to functions that render their inputs safe.
var funcMap = template.FuncMap{
	"exp_template_html_cssescaper":      cssEscaper,
	"exp_template_html_cssvaluefilter":  cssValueFilter,
	"exp_template_html_jsregexpescaper": jsRegexpEscaper,
	"exp_template_html_jsstrescaper":    jsStrEscaper,
	"exp_template_html_jsvalescaper":    jsValEscaper,
	"exp_template_html_nospaceescaper":  htmlNospaceEscaper,
	"exp_template_html_urlescaper":      urlEscaper,
	"exp_template_html_urlfilter":       urlFilter,
	"exp_template_html_urlnormalizer":   urlNormalizer,
}

// escaper collects type inferences about templates and changes needed to make
// templates injection safe.
type escaper struct {
	// set is the template set being escaped.
	set *template.Set
	// output[templateName] is the output context for a templateName that
	// has been mangled to include its input context.
	output map[string]context
	// derived[c.mangle(name)] maps to a template derived from the template
	// named name templateName for the start context c.
	derived map[string]*template.Template
	// called[templateName] is a set of called mangled template names.
	called map[string]bool
	// actionNodeEdits and templateNodeEdits are the accumulated edits to
	// apply during commit. Such edits are not applied immediately in case
	// a template set executes a given template in different escaping
	// contexts.
	actionNodeEdits   map[*parse.ActionNode][]string
	templateNodeEdits map[*parse.TemplateNode]string
}

// filterFailsafe is an innocuous word that is emitted in place of unsafe values
// by sanitizer functions.  It is not a keyword in any programming language,
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
		return e.escapeText(c, n.Text)
	case *parse.WithNode:
		return e.escapeBranch(c, &n.BranchNode, "with")
	}
	panic("escaping " + n.String() + " is unimplemented")
}

// escapeAction escapes an action template node.
func (e *escaper) escapeAction(c context, n *parse.ActionNode) context {
	s := make([]string, 0, 3)
	switch c.state {
	case stateURL, stateCSSDqStr, stateCSSSqStr, stateCSSDqURL, stateCSSSqURL, stateCSSURL:
		switch c.urlPart {
		case urlPartNone:
			s = append(s, "exp_template_html_urlfilter")
			fallthrough
		case urlPartPreQuery:
			switch c.state {
			case stateCSSDqStr, stateCSSSqStr:
				s = append(s, "exp_template_html_cssescaper")
			case stateCSSDqURL, stateCSSSqURL, stateCSSURL:
				s = append(s, "exp_template_html_urlnormalizer")
			}
		case urlPartQueryOrFrag:
			s = append(s, "exp_template_html_urlescaper")
		case urlPartUnknown:
			return context{
				state:   stateError,
				errLine: n.Line,
				errStr:  fmt.Sprintf("%s appears in an ambiguous URL context", n),
			}
		default:
			panic(c.urlPart.String())
		}
	case stateJS:
		s = append(s, "exp_template_html_jsvalescaper")
		// A slash after a value starts a div operator.
		c.jsCtx = jsCtxDivOp
	case stateJSDqStr, stateJSSqStr:
		s = append(s, "exp_template_html_jsstrescaper")
	case stateJSRegexp:
		s = append(s, "exp_template_html_jsregexpescaper")
	case stateComment, stateJSBlockCmt, stateJSLineCmt, stateCSSBlockCmt, stateCSSLineCmt:
		return context{
			state:   stateError,
			errLine: n.Line,
			errStr:  fmt.Sprintf("%s appears inside a comment", n),
		}
	case stateCSS:
		s = append(s, "exp_template_html_cssvaluefilter")
	case stateText:
		s = append(s, "html")
	}
	switch c.delim {
	case delimNone:
		// No extra-escaping needed for raw text content.
	case delimSpaceOrTagEnd:
		s = append(s, "exp_template_html_nospaceescaper")
	default:
		s = append(s, "html")
	}
	if _, ok := e.actionNodeEdits[n]; ok {
		panic(fmt.Sprintf("node %s shared between templates", n))
	}
	e.actionNodeEdits[n] = s
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
		if s[dups] == (id.Args[0].(*parse.IdentifierNode)).Ident {
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
		i := indexOfStr((id.Args[0].(*parse.IdentifierNode)).Ident, s)
		if i != -1 {
			for _, name := range s[:i] {
				newCmds = append(newCmds, newIdentCmd(name))
			}
			s = s[i+1:]
		}
		newCmds = append(newCmds, id)
	}
	// Create any remaining sanitizers.
	for _, name := range s {
		newCmds = append(newCmds, newIdentCmd(name))
	}
	p.Cmds = newCmds
}

// indexOfStr is the least i such that strs[i] == s or -1 if s is not in strs.
func indexOfStr(s string, strs []string) int {
	for i, t := range strs {
		if s == t {
			return i
		}
	}
	return -1
}

// newIdentCmd produces a command containing a single identifier node.
func newIdentCmd(identifier string) *parse.CommandNode {
	return &parse.CommandNode{
		NodeType: parse.NodeCommand,
		Args:     []parse.Node{parse.NewIdentifier(identifier)},
	}
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

	return context{
		state:   stateError,
		errLine: line,
		errStr:  fmt.Sprintf("{{%s}} branches end in different contexts: %v, %v", nodeName, a, b),
	}
}

// escapeBranch escapes a branch template node: "if", "range" and "with".
func (e *escaper) escapeBranch(c context, n *parse.BranchNode, nodeName string) context {
	c0 := e.escapeList(c, n.List)
	if nodeName == "range" && c0.state != stateError {
		// The "true" branch of a "range" node can execute multiple times.
		// We check that executing n.List once results in the same context
		// as executing n.List twice.
		ae, te := e.actionNodeEdits, e.templateNodeEdits
		e.actionNodeEdits, e.templateNodeEdits = make(map[*parse.ActionNode][]string), make(map[*parse.TemplateNode]string)
		c0 = join(c0, e.escapeList(c0, n.List), n.Line, nodeName)
		e.actionNodeEdits, e.templateNodeEdits = ae, te
		if c0.state == stateError {
			// Make clear that this is a problem on loop re-entry
			// since developers tend to overlook that branch when
			// debugging templates.
			c0.errLine = n.Line
			c0.errStr = "on range loop re-entry: " + c0.errStr
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

// escapeTemplate escapes a {{template}} call node.
func (e *escaper) escapeTemplate(c context, n *parse.TemplateNode) context {
	c, name := e.escapeTree(c, n.Name, n.Line)
	if name != n.Name {
		if _, ok := e.templateNodeEdits[n]; ok {
			panic(fmt.Sprintf("node %s shared between templates", n))
		}
		e.templateNodeEdits[n] = name
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
		return context{
			state:   stateError,
			errStr:  fmt.Sprintf("no such template %s", name),
			errLine: line,
		}, dname
	}
	if dname != name {
		// Use any template derived during an earlier call to EscapeSet
		// with different top level templates, or clone if necessary.
		dt := e.template(dname)
		if dt == nil {
			dt = template.New(dname)
			dt.Tree = &parse.Tree{Name: dname, Root: cloneList(t.Root)}
			e.derived[dname] = dt
		}
		t = dt
	}
	return e.computeOutCtx(c, t), dname
}

// computeOutCtx takes a template and its start context and computes the output
// context while storing any inferences in e.
func (e *escaper) computeOutCtx(c context, t *template.Template) context {
	n := t.Name()
	// We need to assume an output context so that recursive template calls
	// do not infinitely recurse, but instead take the fast path out of
	// escapeTree.
	// Naively assume that the input context is the same as the output.
	// This is true >90% of the time, and does not matter if the template
	// is not reentrant.
	e.output[n] = c
	// Start with a fresh called map so e.called[n] below is true iff t is
	// reentrant.
	called := e.called
	e.called = make(map[string]bool)
	// Propagate context over the body.
	d := e.escapeList(c, t.Tree.Root)
	// If t was called, then our assumption above that e.output[n] = c
	// was incorporated into d, so we have to check that assumption.
	if e.called[n] && d.state != stateError && !c.eq(d) {
		d = context{
			state: stateError,
			// TODO: Find the first node with a line in t.Tree.Root
			errLine: 0,
			errStr:  fmt.Sprintf("cannot compute output context for template %s", n),
		}
		// TODO: If necessary, compute a fixed point by assuming d
		// as the input context, and recursing to escapeList with a 
		// different escaper and seeing if starting at d ends in d.
	}
	for k, v := range e.called {
		called[k] = v
	}
	e.called = called
	return d
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

// escapeText escapes a text template node.
func (e *escaper) escapeText(c context, s []byte) context {
	for len(s) > 0 {
		if c.delim == delimNone {
			c, s = transitionFunc[c.state](c, s)
			continue
		}

		i := bytes.IndexAny(s, delimEnds[c.delim])
		if i == -1 {
			// Remain inside the attribute.
			// Decode the value so non-HTML rules can easily handle
			//     <button onclick="alert(&quot;Hi!&quot;)">
			// without having to entity decode token boundaries.
			d := c.delim
			c.delim = delimNone
			c = e.escapeText(c, []byte(html.UnescapeString(string(s))))
			if c.state != stateError {
				c.delim = d
			}
			return c
		}
		if c.delim != delimSpaceOrTagEnd {
			// Consume any quote.
			i++
		}
		// On exiting an attribute, we discard all state information
		// except the state and element.
		c, s = context{state: stateTag, element: c.element}, s[i:]
	}
	return c
}

// commit applies changes to actions and template calls needed to contextually
// autoescape content and adds any derived templates to the set.
func (e *escaper) commit() {
	for name, _ := range e.output {
		e.template(name).Funcs(funcMap)
	}
	for _, t := range e.derived {
		e.set.Add(t)
	}
	for n, s := range e.actionNodeEdits {
		ensurePipelineContains(n.Pipe, s)
	}
	for n, name := range e.templateNodeEdits {
		n.Name = name
	}
}

// template returns the named template given a mangled template name.
func (e *escaper) template(name string) *template.Template {
	t := e.set.Template(name)
	if t == nil {
		t = e.derived[name]
	}
	return t
}
