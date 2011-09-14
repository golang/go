// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package html is a specialization of package template that automates the
construction of HTML output that is safe against code injection.


Introduction

To use this package, invoke the standard template package to parse a template
set, and then use this package’s EscapeSet function to secure the set.
The arguments to EscapeSet are the template set and the names of all templates
that will be passed to Execute.

    set, err := new(template.Set).Parse(...)
    set, err = EscapeSet(set, "templateName0", ...)

If successful, set will now be injection-safe. Otherwise, the returned set will
be nil and an error, described below, will explain the problem.
If an error is returned, do not use the original set; it is insecure.

The template names do not need to include helper templates but should include
all names x used thus:

    set.Execute(out, x, ...)

EscapeSet modifies the named templates in place to treat data values as plain
text safe for embedding in an HTML document. The escaping is contextual, so
actions can appear within JavaScript, CSS, and URI contexts without introducing'hazards.

The security model used by this package assumes that template authors are
trusted, while Execute's data parameter is not. More details are provided below.

Example

    tmpls, err := new(template.Set).Parse(`{{define "t'}}Hello, {{.}}!{{end}}`)

when used by itself

    tmpls.Execute(out, "t", "<script>alert('you have been pwned')</script>")

produces

    Hello, <script>alert('you have been pwned')</script>!

but after securing with EscapeSet like this,

    tmpls, err := EscapeSet(tmpls, "t")
    tmpls.Execute(out, "t", ...)

produces the safe, escaped HTML output

    Hello, &lt;script&gt;alert('you have been pwned')&lt;/script&gt;!


Contexts

EscapeSet understands HTML, CSS, JavaScript, and URIs. It adds sanitizing
functions to each simple action pipeline, so given the excerpt

  <a href="/search?q={{.}}">{{.}}</a>

EscapeSet will rewrite each {{.}} to add escaping functions where necessary,
in this case,

  <a href="/search?q={{. | urlquery}}">{{. | html}}</a>


Errors

This section describes the errors returned by EscapeSet. Each error is
illustrated by an example that triggers the error, followed by an explanation
of the problem.

Error: "... appears in an ambiguous URL context"
Example:
  <a href="
     {{if .C}}
       /path/
     {{else}}
       /search?q=
     {{end}}
     {{.X}}
  ">
Discussion:
  {{.X}} is in an ambiguous URL context since, depending on {{.C}}, it may be
  either a URL suffix or a query parameter.
  Moving {{.X}} into the condition removes the ambiguity:
  <a href="{{if .C}}/path/{{.X}}{{else}}/search?q={{.X}}">


Error: "... appears inside a comment"
Example:
*/
//  <!-- {{.X}} -->
//  <script>/* {{.X}} */</script>
//  <style>/* {{.X}} */</style>
/*
Discussion:
  {{.X}} appears inside a comment. There is no escaping convention for
  comments. To use IE conditional comments, inject the
  whole comment as a type string (see below).
  To comment out code, break the {{...}}.

Error: "{{if}} branches end in different contexts"
Example:
  {{if .C}}<a href="{{end}}{{.X}}
Discussion:
  EscapeSet statically examines each possible path when it encounters a {{if}},
  {{range}}, or {{with}} to escape any following pipelines. The example is
  ambiguous since {{.X}} might be an HTML text node, or a URL prefix in an
  HTML attribute. EscapeSet needs to understand the context of {{.X}} to escape
  it, but that depends on the run-time value of {{.C}}.

  The problem is usually something like missing quotes or angle brackets, or
  can be avoided by refactoring to put the two contexts into different
  branches of an if, range or with. Adding an {{else}} might help.

  First, look for a bug in your template. Missing quotes or '>' can trigger
  this error.

     {{if .C}}<div ... class="foo>{{end}}  <- No quote after foo

  Second, try refactoring your template.

     {{if .C}}<script>alert({{end}}{{.X}}{{if .C}})</script>{{end}}

     ->

     {{if .C}}<script>alert({{.X}})</script>{{else}}{{.X}}{{end}}

  Third, check for {{range}}s that have no {{else}}

    <a href="/search
      {{range $i, $v := .}}
        {{if $i}}&{{else}}?{{end}}
        v={{$v}}
      {{end}}
      &x={{.X}}
      ">

  looks good, but if {{.}} is empty then the URL is /search&x=...
  where {{.X}} is not guaranteed to be in a URL query.
  EscapeSet cannot prove which {{range}} collections are never non-empty, so
  add an {{else}}

    <a href="{{range ...}}...{{end}}&x={{X}}">

    ->

    <a href="{{range ...}}...{{else}}?{{end}}&x={{.X}}">

  Fourth, contact the mailing list. You may have a useful pattern that
  EscapeSet does not yet support, and we can work with you.


Error: "... ends in a non-text context: ..."
Examples:
  <div
  <div title="no close quote>
  <script>f()
Discussion:
  EscapeSet assumes the ouput is a DocumentFragment of HTML.
  Templates that end without closing tags will trigger this warning.
  Templates that produce incomplete Fragments should not be named
  in the call to EscapeSet.


If you have a helper template in your set that is not meant to produce a
  document fragment, then do not pass its name to EscapeSet(set, ...names).

    {{define "main"}} <script>{{template "helper"}}</script> {{end}}
    {{define "helper"}} document.write(' <div title=" ') {{end}}

  "helper" does not produce a valid document fragment, though it does
  produce a valid JavaScript Program.

"must specify names of top level templates"

  EscapeSet does not assume that all templates in a set produce HTML.
  Some may be helpers that produce snippets of other languages.
  Passing in no template names is most likely an error, so EscapeSet(set) will
  panic.
  If you call EscapeSet with a slice of names, guard it with a len check:

    if len(names) != 0 {
      set, err := EscapeSet(set, ...names)
    }

Error: "no such template ..."
Examples:
   {{define "main"}}<div {{template "attrs"}}>{{end}}
   {{define "attrs"}}href="{{.URL}}"{{end}}
Discussion:
  EscapeSet looks through template calls to compute the context.
  Here the {{.URL}} in "attrs" must be treated as a URL when called from "main",
  but if "attrs" is not in set when EscapeSet(&set, "main") is called, this
  error will arise.

Error: "on range loop re-entry: ..."
Example:
  {{range .}}<p class={{.}}{{end}}
Discussion:
  If an iteration through a range would cause it to end in
  a different context than an earlier pass, there is no single context.
  In the example, the <p> tag is missing a '>'.
  EscapeSet cannot tell whether {{.}} is meant to be an HTML class or the
  content of a broken <p> element and complains because the second iteration
  would produce something like

    <p class=foo<p class=bar

Error: "unfinished escape sequence in ..."
Example:
  <script>alert("\{{.X}}")</script>
Discussion:
  EscapeSet does not support actions following a backslash.
  This is usually an error and there are better solutions; for
  our example
    <script>alert("{{.X}}")</script>
  should work, and if {{.X}} is a partial escape sequence such as
  "xA0", give it the type ContentTypeJSStr and include the whole
  sequence, as in
    {`\xA0`, ContentTypeJSStr}

Error: "unfinished JS regexp charset in ..."
Example:
    <script>var pattern = /foo[{{.Chars}}]/</script>
Discussion:
  EscapeSet does not support interpolation into regular expression literal
  character sets.

Error: "ZgotmplZ"
Example:
  <img src="{{.X}}">
  where {{.X}} evaluates to `javascript:...`
Discussion:
  "ZgotmplZ" is a special value that indicates that unsafe content reached
  a CSS or URL context at runtime. The output of the example will be
    <img src="#ZgotmplZ">
  If the data can be trusted, giving the string type XXX will exempt
  it from filtering.

A fuller picture

The rest of this package comment may be skipped on first reading; it includes
details necessary to understand escaping contexts and error messages. Most users
will not need to understand these details.




Contexts

Assuming {{.}} is `O'Reilly: How are <i>you</i>?`, the table below shows
how {{.}} appears when used in the context to the left.

Context                          {{.}} After
{{.}}                            O'Reilly: How are &lt;i&gt;you&lt;/i&gt;?
<a title='{{.}}'>                O&#39;Reilly: How are you?
<a href="/{{.}}">                O&#39;Reilly: How are %3ci%3eyou%3c/i%3e?
<a href="?q={{.}}">              O&#39;Reilly%3a%20How%20are%3ci%3e...%3f
<a onx='f("{{.}}")'>             O\x27Reilly: How are \x3ci\x3eyou...?
<a onx='f({{.}})'>               "O\x27Reilly: How are \x3ci\x3eyou...?"
<a onx='pattern = /{{.}}/;'>     O\x27Reilly: How are \x3ci\x3eyou...\x3f

If used in an unsafe context, then the value might be filtered out:

Context                          {{.}} After
<a href="{{.}}">                 #ZgotmplZ

since "O'Reilly:" is not an allowed protocol like "http:".


If {{.}} is the innocuous word, `left`, then it can appear more widely,

Context                              {{.}} After
{{.}}                                left
<a title='{{.}}'>                    left
<a href='{{.}}'>                     left
<a href='/{{.}}'>                    left
<a href='?dir={{.}}'>                left
<a style="border-{{.}}: 4px">        left
<a style="align: {{.}}">             left
<a style="background: '{{.}}'>       left
<a style="background: url('{{.}}')>  left
<style>p.{{.}} {color:red}</style>   left

Non-string values can be used in JavaScript contexts.
If {{.}} is

  []struct{A,B string}{ "foo", "bar" }

in the escaped template

  <script>var pair = {{.}};</script>

then the template output is

  <script>var pair = {"A": "foo", "B": "bar"};</script>

See package json to understand how non-string content is marshalled for
embedding in JavaScript contexts.


Typed Strings

By default, EscapeSet assumes all pipelines produce a plain text string. It
adds escaping pipeline stages necessary to correctly and safely embed that
plain text string in the appropriate context.

When a data value is not plain text, you can make sure it is not over-escaped
by marking it with its type.

A value that implements interface TypedStringer can carry known-safe content.

  type safeHTML struct{}
  func (s safeHTML) String() string { return `<b>World</b>` }
  func (s safeHTML) ContentType() ContentType { return ContentTypeHTML }

The template

  Hello, {{.}}!

can be invoked with

  tmpl.Execute(out, safeHTML{})

to produce

  Hello, <b>World</b>!

instead of the

  Hello, &lt;b&gt;World&lt;b&gt;!

which would have been produced if {{.}} did not implement TypedStringer.

ContentTypeHTML attaches to a well-formed HTML DocumentFragment.
Do not use it for HTML from a third-party, or HTML with unclosed tags or
comments. The outputs of a sound HTML sanitizer and a template escaped by
this package are examples of ContentTypeHTML.

ContentTypeCSS attaches to a well-formed safe content that matches:
(1) The CSS3 stylesheet production, for example `p { color: purple }`
(2) The CSS3 rule production, for example `a[href=~"https:"].foo#bar`
(3) CSS3 declaration productions, for example `color: red; margin: 2px`
(4) The CSS3 value production, for example `rgba(0, 0, 255, 127)`

ContentTypeJS attaches to a well-formed JavaScript (EcmaScript5) Expression
production, for example `(x + y * z())`. Template authors are responsible
for ensuring that typed expressions do not break the intended precedence and
that there is no statement/expression ambiguity as when passing an expression
like "{ foo: bar() }\n['foo']()" which is both a valid Expression and a valid
Program with a very different meaning.

ContentTypeJSStr attaches to a snippet of \-escaped characters that could be
quoted to form a JavaScript string literal. For example, foo\nbar with quotes
around it makes a valid JavaScript string literal.

ContentTypeURL attaches to a URL fragment from a trusted source.
A URL like `javascript:checkThatFormNotEditedBeforeLeavingPage()`
from a trusted source should go in the page, but by default dynamic
`javascript:` URLs are filtered out since they are a frequently
successfully exploited injection vector.


Security Model

http://js-quasis-libraries-and-repl.googlecode.com/svn/trunk/safetemplate.html#problem_definition defines "safe" as used by this package.

This package assumes that template authors are trusted, that Execute's data
parameter is not, and seeks to preserve the properties below in the face
of untrusted data:

Structure Preservation Property
"... when a template author writes an HTML tag in a safe templating language,
the browser will interpret the corresponding portion of the output as a tag
regardless of the values of untrusted data, and similarly for other structures
such as attribute boundaries and JS and CSS string boundaries."

Code Effect Property
"... only code specified by the template author should run as a result of
injecting the template output into a page and all code specified by the
template author should run as a result of the same."

Least Surprise Property
"A developer (or code reviewer) familiar with HTML, CSS, and JavaScript;
who knows that EscapeSet is applied should be able to look at a {{.}}
and correctly infer what sanitization happens."
*/
package html
