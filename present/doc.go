// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package present implements parsing and rendering of present files,
which can be slide presentations as in golang.org/x/tools/cmd/present
or articles as in golang.org/x/blog (the Go blog).

File Format

Present files begin with a header giving the title of the document
and other metadata, which looks like:

	# Title of document
	Subtitle of document
	15:04 2 Jan 2006
	Tags: foo, bar, baz
	Summary: This is a great document you want to read.
	OldURL: former-path-for-this-doc

The "# " prefix before the title indicates that this is
a Markdown-enabled present file: it uses
Markdown for text markup in the body of the file.
If the "# " prefix is missing, the file uses
legacy present markup, described below.

The date line may be written without a time:
	2 Jan 2006
In this case, the time will be interpreted as 10am UTC on that date.

The tags line is a comma-separated list of tags that may be used to categorize
the document.

The summary line gives a short summary used in blog feeds.

The old URL line, which may be repeated, gives an older (perhaps relative) URL
for this document.
A server might use these to generate appropriate redirects.

Only the title is required;
the subtitle, date, tags, summary, and old URL lines are optional.
In Markdown-enabled present, the summary defaults to being empty.
In legacy present, the summary defaults to the first paragraph of text.

After the header come zero or more author blocks, like this:

	Author Name
	Job title, Company
	joe@example.com
	https://url/
	@twitter_name

The first line of the author block is conventionally the author name.
Otherwise, the author section may contain a mixture of text, twitter names, and links.
For slide presentations, only the plain text lines will be displayed on the
first slide.

If multiple author blocks are listed, each new block must be preceded
by its own blank line.

After the author blocks come the presentation slides or article sections,
which can in turn have subsections.
In Markdown-enabled present files, each slide or section begins with a "##" header line,
subsections begin with a "###" header line, and so on.
In legacy present files, each slide or section begins with a "*" header line,
subsections begin with a "**" header line, and so on.

In addition to the marked-up text in a section (or subsection),
a present file can contain present command invocations, each of which begins
with a dot, as in:

	.code x.go /^func main/,/^}/
	.play y.go
	.image image.jpg
	.background image.jpg
	.iframe https://foo
	.link https://foo label
	.html file.html
	.caption _Gopher_ by [[https://instagram.com/reneefrench][Renee French]]

Other than the commands, the text in a section is interpreted
either as Markdown or as legacy present markup.

Markdown Syntax

Markdown typically means the generic name for a family of similar markup languages.
The specific variant used in present is CommonMark.
See https://commonmark.org/help/tutorial/ for a quick tutorial.

In Markdown-enabled present,
section headings can end in {#name} to set the HTML anchor ID for the heading to "name".

Lines beginning with "//" (outside of code blocks, of course)
are treated as present comments and have no effect.

Lines beginning with ": " are treated as speaker notes, described below.

Example:

	# Title of Talk

	My Name
	9 Mar 2020
	me@example.com

	## Title of Slide or Section (must begin with ##)

	Some Text

	### Subsection {#anchor}

	- bullets
	- more bullets
	- a bullet continued
	  on the next line

	#### Sub-subsection

	Some More text

		Preformatted text (code block)
		is indented (by one tab, or four spaces)

	Further Text, including command invocations.

	## Section 2: Example formatting {#fmt}

	Formatting:

	_italic_
	// A comment that is completely ignored.
	: Speaker notes.
	**bold**
	`program`
	Markup—_especially italic text_—can easily be overused.
	_Why use scoped\_ptr_? Use plain **\*ptr** instead.

	Visit [the Go home page](https://golang.org/).

Legacy Present Syntax

Compared to Markdown,
in legacy present
slides/sections use "*" instead of "##",
whole-line comments begin with "#" instead of "//",
bullet lists can only contain single (possibly wrapped) text lines,
and the font styling and link syntaxes are subtly different.

Example:

	Title of Talk

	My Name
	1 Jan 2013
	me@example.com

	* Title of Slide or Section (must begin with *)

	Some Text

	** Subsection

	- bullets
	- more bullets
	- a bullet continued
	  on the next line (indented at least one space)

	*** Sub-subsection

	Some More text

	  Preformatted text (code block)
	  is indented (however you like)

	Further Text, including command invocations.

	* Section 2: Example formatting

	Formatting:

	_italic_
	*bold*
	`program`
	Markup—_especially_italic_text_—can easily be overused.
	_Why_use_scoped__ptr_? Use plain ***ptr* instead.

	Visit [[https://golang.org][the Go home page]].

Within the input for plain text or lists, text bracketed by font
markers will be presented in italic, bold, or program font.
Marker characters are _ (italic), * (bold) and ` (program font).
An opening marker must be preceded by a space or punctuation
character or else be at start of a line; similarly, a closing
marker must be followed by a space or punctuation character or
else be at the end of a line. Unmatched markers appear as plain text.
There must be no spaces between markers. Within marked text,
a single marker character becomes a space and a doubled single
marker quotes the marker character.

Links can be included in any text with the form [[url][label]], or
[[url]] to use the URL itself as the label.

Command Invocations

A number of special commands are available through invocations
in the input text. Each such invocation contains a period as the
first character on the line, followed immediately by the name of
the function, followed by any arguments. A typical invocation might
be

	.play demo.go /^func show/,/^}/

(except that the ".play" must be at the beginning of the line and
not be indented as in this comment.)

Here follows a description of the functions:

code:

Injects program source into the output by extracting code from files
and injecting them as HTML-escaped <pre> blocks.  The argument is
a file name followed by an optional address that specifies what
section of the file to display. The address syntax is similar in
its simplest form to that of ed, but comes from sam and is more
general. See
	https://plan9.io/sys/doc/sam/sam.html Table II
for full details. The displayed block is always rounded out to a
full line at both ends.

If no pattern is present, the entire file is displayed.

Any line in the program that ends with the four characters
	OMIT
is deleted from the source before inclusion, making it easy
to write things like
	.code test.go /START OMIT/,/END OMIT/
to find snippets like this
	tedious_code = boring_function()
	// START OMIT
	interesting_code = fascinating_function()
	// END OMIT
and see only this:
	interesting_code = fascinating_function()

Also, inside the displayed text a line that ends
	// HL
will be highlighted in the display. A highlighting mark may have a
suffix word, such as
	// HLxxx
Such highlights are enabled only if the code invocation ends with
"HL" followed by the word:
	.code test.go /^type Foo/,/^}/ HLxxx

The .code function may take one or more flags immediately preceding
the filename. This command shows test.go in an editable text area:
	.code -edit test.go
This command shows test.go with line numbers:
	.code -numbers test.go

play:

The function "play" is the same as "code" but puts a button
on the displayed source so the program can be run from the browser.
Although only the selected text is shown, all the source is included
in the HTML output so it can be presented to the compiler.

link:

Create a hyperlink. The syntax is 1 or 2 space-separated arguments.
The first argument is always the HTTP URL.  If there is a second
argument, it is the text label to display for this link.

	.link https://golang.org golang.org

image:

The template uses the function "image" to inject picture files.

The syntax is simple: 1 or 3 space-separated arguments.
The first argument is always the file name.
If there are more arguments, they are the height and width;
both must be present, or substituted with an underscore.
Replacing a dimension argument with the underscore parameter
preserves the aspect ratio of the image when scaling.

	.image images/betsy.jpg 100 200
	.image images/janet.jpg _ 300

video:

The template uses the function "video" to inject video files.

The syntax is simple: 2 or 4 space-separated arguments.
The first argument is always the file name.
The second argument is always the file content-type.
If there are more arguments, they are the height and width;
both must be present, or substituted with an underscore.
Replacing a dimension argument with the underscore parameter
preserves the aspect ratio of the video when scaling.

	.video videos/evangeline.mp4 video/mp4 400 600

	.video videos/mabel.ogg video/ogg 500 _

background:

The template uses the function "background" to set the background image for
a slide.  The only argument is the file name of the image.

	.background images/susan.jpg

caption:

The template uses the function "caption" to inject figure captions.

The text after ".caption" is embedded in a figcaption element after
processing styling and links as in standard text lines.

	.caption _Gopher_ by [[https://instagram.com/reneefrench][Renee French]]

iframe:

The function "iframe" injects iframes (pages inside pages).
Its syntax is the same as that of image.

html:

The function html includes the contents of the specified file as
unescaped HTML. This is useful for including custom HTML elements
that cannot be created using only the slide format.
It is your responsibility to make sure the included HTML is valid and safe.

	.html file.html

Presenter Notes

Lines that begin with ": " are treated as presenter notes,
in both Markdown and legacy present syntax.
By default, presenter notes are collected but ignored.

When running the present command with -notes,
typing 'N' in your browser displaying your slides
will create a second window displaying the notes.
The second window is completely synced with the main
window, except that presenter notes are only visible in the second window.

Notes may appear anywhere within the slide text. For example:

	* Title of slide

	Some text.

	: Presenter notes (first paragraph)

	Some more text.

	: Presenter notes (subsequent paragraph(s))

*/
package present // import "golang.org/x/tools/present"
