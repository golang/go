// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package template (html/template) implements data-driven templates for
generating HTML output safe against code injection. It provides the
same interface as [text/template] and should be used instead of
[text/template] whenever the output is HTML.

The documentation here focuses on the security features of the package.
For information about how to program the templates themselves, see the
documentation for [text/template].

# Introduction

This package wraps [text/template] so you can share its template API
to parse and execute HTML templates safely.

	tmpl, err := template.New("name").Parse(...)
	// Error checking elided
	err = tmpl.Execute(out, data)

If successful, tmpl will now be injection-safe. Otherwise, err is an error
defined in the docs for ErrorCode.

HTML templates treat data values as plain text which should be encoded so they
can be safely embedded in an HTML document. The escaping is contextual, so
actions can appear within JavaScript, CSS, and URI contexts.

The security model used by this package assumes that template authors are
trusted, while Execute's data parameter is not. More details are
provided below.

Example

	import "text/template"
	...
	t, err := template.New("foo").Parse(`{{define "T"}}Hello, {{.}}!{{end}}`)
	err = t.ExecuteTemplate(out, "T", "<script>alert('you have been pwned')</script>")

produces

	Hello, <script>alert('you have been pwned')</script>!

but the contextual autoescaping in html/template

	import "html/template"
	...
	t, err := template.New("foo").Parse(`{{define "T"}}Hello, {{.}}!{{end}}`)
	err = t.ExecuteTemplate(out, "T", "<script>alert('you have been pwned')</script>")

produces safe, escaped HTML output

	Hello, &lt;script&gt;alert(&#39;you have been pwned&#39;)&lt;/script&gt;!

# Contexts

This package understands HTML, CSS, JavaScript, and URIs. It adds sanitizing
functions to each simple action pipeline, so given the excerpt

	<a href="/search?q={{.}}">{{.}}</a>

At parse time each {{.}} is overwritten to add escaping functions as necessary.
In this case it becomes

	<a href="/search?q={{. | urlescaper | attrescaper}}">{{. | htmlescaper}}</a>

where urlescaper, attrescaper, and htmlescaper are aliases for internal escaping
functions.

For these internal escaping functions, if an action pipeline evaluates to
a nil interface value, it is treated as though it were an empty string.

# Namespaced and data- attributes

Attributes with a namespace are treated as if they had no namespace.
Given the excerpt

	<a my:href="{{.}}"></a>

At parse time the attribute will be treated as if it were just "href".
So at parse time the template becomes:

	<a my:href="{{. | urlescaper | attrescaper}}"></a>

Similarly to attributes with namespaces, attributes with a "data-" prefix are
treated as if they had no "data-" prefix. So given

	<a data-href="{{.}}"></a>

At parse time this becomes

	<a data-href="{{. | urlescaper | attrescaper}}"></a>

If an attribute has both a namespace and a "data-" prefix, only the namespace
will be removed when determining the context. For example

	<a my:data-href="{{.}}"></a>

This is handled as if "my:data-href" was just "data-href" and not "href" as
it would be if the "data-" prefix were to be ignored too. Thus at parse
time this becomes just

	<a my:data-href="{{. | attrescaper}}"></a>

As a special case, attributes with the namespace "xmlns" are always treated
as containing URLs. Given the excerpts

	<a xmlns:title="{{.}}"></a>
	<a xmlns:href="{{.}}"></a>
	<a xmlns:onclick="{{.}}"></a>

At parse time they become:

	<a xmlns:title="{{. | urlescaper | attrescaper}}"></a>
	<a xmlns:href="{{. | urlescaper | attrescaper}}"></a>
	<a xmlns:onclick="{{. | urlescaper | attrescaper}}"></a>

# Errors

See the documentation of ErrorCode for details.

# A fuller picture

The rest of this package comment may be skipped on first reading; it includes
details necessary to understand escaping contexts and error messages. Most users
will not need to understand these details.

# Contexts

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

	struct{A,B string}{ "foo", "bar" }

in the escaped template

	<script>var pair = {{.}};</script>

then the template output is

	<script>var pair = {"A": "foo", "B": "bar"};</script>

See package json to understand how non-string content is marshaled for
embedding in JavaScript contexts.

# Typed Strings

By default, this package assumes that all pipelines produce a plain text string.
It adds escaping pipeline stages necessary to correctly and safely embed that
plain text string in the appropriate context.

When a data value is not plain text, you can make sure it is not over-escaped
by marking it with its type.

Types HTML, JS, URL, and others from content.go can carry safe content that is
exempted from escaping.

The template

	Hello, {{.}}!

can be invoked with

	tmpl.Execute(out, template.HTML(`<b>World</b>`))

to produce

	Hello, <b>World</b>!

instead of the

	Hello, &lt;b&gt;World&lt;b&gt;!

that would have been produced if {{.}} was a regular string.

# Security Model

https://rawgit.com/mikesamuel/sanitized-jquery-templates/trunk/safetemplate.html#problem_definition defines "safe" as used by this package.

This package assumes that template authors are trusted, that Execute's data
parameter is not, and seeks to preserve the properties below in the face
of untrusted data:

Structure Preservation Property:
"... when a template author writes an HTML tag in a safe templating language,
the browser will interpret the corresponding portion of the output as a tag
regardless of the values of untrusted data, and similarly for other structures
such as attribute boundaries and JS and CSS string boundaries."

Code Effect Property:
"... only code specified by the template author should run as a result of
injecting the template output into a page and all code specified by the
template author should run as a result of the same."

Least Surprise Property:
"A developer (or code reviewer) familiar with HTML, CSS, and JavaScript, who
knows that contextual autoescaping happens should be able to look at a {{.}}
and correctly infer what sanitization happens."

Previously, ECMAScript 6 template literal were disabled by default, and could be
enabled with the GODEBUG=jstmpllitinterp=1 environment variable. Template
literals are now supported by default, and setting jstmpllitinterp has no
effect.
*/
package template
