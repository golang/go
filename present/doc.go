// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
The present file format

Present files have the following format.  The first non-blank non-comment
line is the title, so the header looks like

	Title of document
	Subtitle of document
	15:04 2 Jan 2006
	Tags: foo, bar, baz
	<blank line>
	Author Name
	Job title, Company
	joe@example.com
	http://url/
	@twitter_name

The subtitle, date, and tags lines are optional.

The date line may be written without a time:
	2 Jan 2006
In this case, the time will be interpreted as 10am UTC on that date.

The tags line is a comma-separated list of tags that may be used to categorize
the document.

The author section may contain a mixture of text, twitter names, and links.
For slide presentations, only the plain text lines will be displayed on the
first slide.

Multiple presenters may be specified, separated by a blank line.

After that come slides/sections, each after a blank line:

	* Title of slide or section (must have asterisk)

	Some Text

	** Subsection

	- bullets
	- more bullets
	- a bullet with

	*** Sub-subsection

	Some More text

	  Preformatted text
	  is indented (however you like)

	Further Text, including invocations like:

	.code x.go /^func main/,/^}/
	.play y.go
	.image image.jpg
	.background image.jpg
	.iframe http://foo
	.link http://foo label
	.html file.html
	.caption _Gopher_ by [[https://www.instagram.com/reneefrench/][Renée French]]

	Again, more text

Blank lines are OK (not mandatory) after the title and after the
text.  Text, bullets, and .code etc. are all optional; title is
not.

Lines starting with # in column 1 are commentary.

Fonts:

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

	_italic_
	*bold*
	`program`
	Markup—_especially_italic_text_—can easily be overused.
	_Why_use_scoped__ptr_? Use plain ***ptr* instead.

Inline links:

Links can be included in any text with the form [[url][label]], or
[[url]] to use the URL itself as the label.

Functions:

A number of template functions are available through invocations
in the input text. Each such invocation contains a period as the
first character on the line, followed immediately by the name of
the function, followed by any arguments. A typical invocation might
be
	.play demo.go /^func show/,/^}/
(except that the ".play" must be at the beginning of the line and
not be indented like this.)

Here follows a description of the functions:

code:

Injects program source into the output by extracting code from files
and injecting them as HTML-escaped <pre> blocks.  The argument is
a file name followed by an optional address that specifies what
section of the file to display. The address syntax is similar in
its simplest form to that of ed, but comes from sam and is more
general. See
	http://plan9.bell-labs.com/sys/doc/sam/sam.html Table II
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
will be highlighted in the display; the 'h' key in the browser will
toggle extra emphasis of any highlighted lines. A highlighting mark
may have a suffix word, such as
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

	.link http://golang.org golang.org

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

	.caption _Gopher_ by [[http://www.reneefrench.com][Renée French]]

iframe:

The function "iframe" injects iframes (pages inside pages).
Its syntax is the same as that of image.

html:

The function html includes the contents of the specified file as
unescaped HTML. This is useful for including custom HTML elements
that cannot be created using only the slide format.
It is your responsibilty to make sure the included HTML is valid and safe.

	.html file.html

Presenter notes:

Presenter notes may be enabled by appending the "-notes" flag when you run
your "present" binary.

This will allow you to open a second window by pressing 'N' from your browser
displaying your slides. The second window is completely synced with your main
window, except that presenter notes are only visible on the second window.

Lines that begin with ": " are treated as presenter notes.

	* Title of slide

	Some Text

	: Presenter notes (first paragraph)
	: Presenter notes (subsequent paragraph(s))

Notes may appear anywhere within the slide text. For example:

	* Title of slide

	: Presenter notes (first paragraph)

	Some Text

	: Presenter notes (subsequent paragraph(s))

This has the same result as the example above.

*/
package present // import "golang.org/x/tools/present"
