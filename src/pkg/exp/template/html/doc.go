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

See the documentation of ErrorCode for details.


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

Types HTML, JS, URL, and others from content.go can carry safe content that is
exempted from escaping.

The template

  Hello, {{.}}!

can be invoked with

  tmpl.Execute(out, HTML(`<b>World</b>`))

to produce

  Hello, <b>World</b>!

instead of the

  Hello, &lt;b&gt;World&lt;b&gt;!

that would have been produced if {{.}} was a regular string.


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
