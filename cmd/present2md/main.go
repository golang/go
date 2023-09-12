// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Present2md converts legacy-syntax present files to Markdown-syntax present files.
//
// Usage:
//
//	present2md [-w] [file ...]
//
// By default, present2md prints the Markdown-syntax form of each input file to standard output.
// If no input file is listed, standard input is used.
//
// The -w flag causes present2md to update the files in place, overwriting each with its
// Markdown-syntax equivalent.
//
// Examples
//
//	present2md your.article
//	present2md -w *.article
package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"log"
	"net/url"
	"os"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/tools/present"
)

func usage() {
	fmt.Fprintf(os.Stderr, "usage: present2md [-w] [file ...]\n")
	os.Exit(2)
}

var (
	writeBack  = flag.Bool("w", false, "write conversions back to original files")
	exitStatus = 0
)

func main() {
	log.SetPrefix("present2md: ")
	log.SetFlags(0)
	flag.Usage = usage
	flag.Parse()

	args := flag.Args()
	if len(args) == 0 {
		if *writeBack {
			log.Fatalf("cannot use -w with standard input")
		}
		convert(os.Stdin, "stdin", false)
		return
	}

	for _, arg := range args {
		f, err := os.Open(arg)
		if err != nil {
			log.Print(err)
			exitStatus = 1
			continue
		}
		err = convert(f, arg, *writeBack)
		f.Close()
		if err != nil {
			log.Print(err)
			exitStatus = 1
		}
	}
	os.Exit(exitStatus)
}

// convert reads the data from r, parses it as legacy present,
// and converts it to Markdown-enabled present.
// If any errors occur, the data is reported as coming from file.
// If writeBack is true, the converted version is written back to file.
// If writeBack is false, the converted version is printed to standard output.
func convert(r io.Reader, file string, writeBack bool) error {
	data, err := io.ReadAll(r)
	if err != nil {
		return err
	}
	if bytes.HasPrefix(data, []byte("# ")) {
		return fmt.Errorf("%v: already markdown", file)
	}

	// Convert all comments before parsing the document.
	// The '//' comment is treated as normal text and so
	// is passed through the translation unaltered.
	data = bytes.Replace(data, []byte("\n#"), []byte("\n//"), -1)

	doc, err := present.Parse(bytes.NewReader(data), file, 0)
	if err != nil {
		return err
	}

	// Title and Subtitle, Time, Tags.
	var md bytes.Buffer
	fmt.Fprintf(&md, "# %s\n", doc.Title)
	if doc.Subtitle != "" {
		fmt.Fprintf(&md, "%s\n", doc.Subtitle)
	}
	if !doc.Time.IsZero() {
		fmt.Fprintf(&md, "%s\n", doc.Time.Format("2 Jan 2006"))
	}
	if len(doc.Tags) > 0 {
		fmt.Fprintf(&md, "Tags: %s\n", strings.Join(doc.Tags, ", "))
	}

	// Summary, defaulting to first paragraph of section.
	// (Summaries must be explicit for Markdown-enabled present,
	// and the expectation is that they will be shorter than the
	// whole first paragraph. But this is what the blog does today.)
	if strings.HasSuffix(file, ".article") && len(doc.Sections) > 0 {
		for _, elem := range doc.Sections[0].Elem {
			text, ok := elem.(present.Text)
			if !ok || text.Pre {
				// skip everything but non-text elements
				continue
			}
			fmt.Fprintf(&md, "Summary:")
			for i, line := range text.Lines {
				fmt.Fprintf(&md, " ")
				printStyled(&md, line, i == 0)
			}
			fmt.Fprintf(&md, "\n")
			break
		}
	}

	// Authors
	for _, a := range doc.Authors {
		fmt.Fprintf(&md, "\n")
		for _, elem := range a.Elem {
			switch elem := elem.(type) {
			default:
				// Can only happen if this type switch is incomplete, which is a bug.
				log.Fatalf("%s: unexpected author type %T", file, elem)
			case present.Text:
				for _, line := range elem.Lines {
					fmt.Fprintf(&md, "%s\n", markdownEscape(line, true))
				}
			case present.Link:
				fmt.Fprintf(&md, "%s\n", markdownEscape(elem.Label, true))
			}
		}
	}

	// Invariant: the output ends in non-blank line now,
	// and after printing any piece of the file below,
	// the output should still end in a non-blank line.
	// If a blank line separator is needed, it should be printed
	// before the block that needs separating, not after.

	if len(doc.TitleNotes) > 0 {
		fmt.Fprintf(&md, "\n")
		for _, line := range doc.TitleNotes {
			fmt.Fprintf(&md, ": %s\n", line)
		}
	}

	if len(doc.Sections) == 1 && strings.HasSuffix(file, ".article") {
		// Blog drops section headers when there is only one section.
		// Don't print a title in this case, to make clear that it's being dropped.
		fmt.Fprintf(&md, "\n##\n")
		printSectionBody(file, 1, &md, doc.Sections[0].Elem)
	} else {
		for _, s := range doc.Sections {
			fmt.Fprintf(&md, "\n")
			fmt.Fprintf(&md, "## %s\n", markdownEscape(s.Title, false))
			printSectionBody(file, 1, &md, s.Elem)
		}
	}

	if !writeBack {
		os.Stdout.Write(md.Bytes())
		return nil
	}
	return os.WriteFile(file, md.Bytes(), 0666)
}

func printSectionBody(file string, depth int, w *bytes.Buffer, elems []present.Elem) {
	for _, elem := range elems {
		switch elem := elem.(type) {
		default:
			// Can only happen if this type switch is incomplete, which is a bug.
			log.Fatalf("%s: unexpected present element type %T", file, elem)

		case present.Text:
			fmt.Fprintf(w, "\n")
			lines := elem.Lines
			for len(lines) > 0 && lines[0] == "" {
				lines = lines[1:]
			}
			if elem.Pre {
				for _, line := range strings.Split(strings.TrimRight(elem.Raw, "\n"), "\n") {
					if line == "" {
						fmt.Fprintf(w, "\n")
					} else {
						fmt.Fprintf(w, "\t%s\n", line)
					}
				}
			} else {
				for _, line := range elem.Lines {
					printStyled(w, line, true)
					fmt.Fprintf(w, "\n")
				}
			}

		case present.List:
			fmt.Fprintf(w, "\n")
			for _, item := range elem.Bullet {
				fmt.Fprintf(w, "  - ")
				for i, line := range strings.Split(item, "\n") {
					if i > 0 {
						fmt.Fprintf(w, "    ")
					}
					printStyled(w, line, false)
					fmt.Fprintf(w, "\n")
				}
			}

		case present.Section:
			fmt.Fprintf(w, "\n")
			sep := " "
			if elem.Title == "" {
				sep = ""
			}
			fmt.Fprintf(w, "%s%s%s\n", strings.Repeat("#", depth+2), sep, markdownEscape(elem.Title, false))
			printSectionBody(file, depth+1, w, elem.Elem)

		case interface{ PresentCmd() string }:
			// If there are multiple present commands in a row, don't print a blank line before the second etc.
			b := w.Bytes()
			sep := "\n"
			if len(b) > 0 {
				i := bytes.LastIndexByte(b[:len(b)-1], '\n')
				if b[i+1] == '.' {
					sep = ""
				}
			}
			fmt.Fprintf(w, "%s%s\n", sep, elem.PresentCmd())
		}
	}
}

func markdownEscape(s string, startLine bool) string {
	var b strings.Builder
	for i, r := range s {
		switch {
		case r == '#' && i == 0,
			r == '*',
			r == '_',
			r == '<' && (i == 0 || s[i-1] != ' ') && i+1 < len(s) && s[i+1] != ' ',
			r == '[' && strings.Contains(s[i:], "]("):
			b.WriteRune('\\')
		}
		b.WriteRune(r)
	}
	return b.String()
}

// Copy of ../../present/style.go adjusted to produce Markdown instead of HTML.

/*
	Fonts are demarcated by an initial and final char bracketing a
	space-delimited word, plus possibly some terminal punctuation.
	The chars are
		_ for italic
		* for bold
		` (back quote) for fixed width.
	Inner appearances of the char become spaces. For instance,
		_this_is_italic_!
	becomes
		<i>this is italic</i>!
*/

func printStyled(w *bytes.Buffer, text string, startLine bool) {
	w.WriteString(font(text, startLine))
}

// font returns s with font indicators turned into HTML font tags.
func font(s string, startLine bool) string {
	if !strings.ContainsAny(s, "[`_*") {
		return markdownEscape(s, startLine)
	}
	words := split(s)
	var b bytes.Buffer
Word:
	for w, word := range words {
		words[w] = markdownEscape(word, startLine && w == 0) // for all the continue Word
		if len(word) < 2 {
			continue Word
		}
		if link, _ := parseInlineLink(word); link != "" {
			words[w] = link
			continue Word
		}
		const marker = "_*`"
		// Initial punctuation is OK but must be peeled off.
		first := strings.IndexAny(word, marker)
		if first == -1 {
			continue Word
		}
		// Opening marker must be at the beginning of the token or else preceded by punctuation.
		if first != 0 {
			r, _ := utf8.DecodeLastRuneInString(word[:first])
			if !unicode.IsPunct(r) {
				continue Word
			}
		}
		open, word := markdownEscape(word[:first], startLine && w == 0), word[first:]
		char := word[0] // ASCII is OK.
		close := ""
		switch char {
		default:
			continue Word
		case '_':
			open += "_"
			close = "_"
		case '*':
			open += "**"
			close = "**"
		case '`':
			open += "`"
			close = "`"
		}
		// Closing marker must be at the end of the token or else followed by punctuation.
		last := strings.LastIndex(word, word[:1])
		if last == 0 {
			continue Word
		}
		if last+1 != len(word) {
			r, _ := utf8.DecodeRuneInString(word[last+1:])
			if !unicode.IsPunct(r) {
				continue Word
			}
		}
		head, tail := word[:last+1], word[last+1:]
		b.Reset()
		var wid int
		for i := 1; i < len(head)-1; i += wid {
			var r rune
			r, wid = utf8.DecodeRuneInString(head[i:])
			if r != rune(char) {
				// Ordinary character.
				b.WriteRune(r)
				continue
			}
			if head[i+1] != char {
				// Inner char becomes space.
				b.WriteRune(' ')
				continue
			}
			// Doubled char becomes real char.
			// Not worth worrying about "_x__".
			b.WriteByte(char)
			wid++ // Consumed two chars, both ASCII.
		}
		text := b.String()
		if close == "`" {
			for strings.Contains(text, close) {
				open += "`"
				close += "`"
			}
		} else {
			text = markdownEscape(text, false)
		}
		words[w] = open + text + close + tail
	}
	return strings.Join(words, "")
}

// split is like strings.Fields but also returns the runs of spaces
// and treats inline links as distinct words.
func split(s string) []string {
	var (
		words = make([]string, 0, 10)
		start = 0
	)

	// appendWord appends the string s[start:end] to the words slice.
	// If the word contains the beginning of a link, the non-link portion
	// of the word and the entire link are appended as separate words,
	// and the start index is advanced to the end of the link.
	appendWord := func(end int) {
		if j := strings.Index(s[start:end], "[["); j > -1 {
			if _, l := parseInlineLink(s[start+j:]); l > 0 {
				// Append portion before link, if any.
				if j > 0 {
					words = append(words, s[start:start+j])
				}
				// Append link itself.
				words = append(words, s[start+j:start+j+l])
				// Advance start index to end of link.
				start = start + j + l
				return
			}
		}
		// No link; just add the word.
		words = append(words, s[start:end])
		start = end
	}

	wasSpace := false
	for i, r := range s {
		isSpace := unicode.IsSpace(r)
		if i > start && isSpace != wasSpace {
			appendWord(i)
		}
		wasSpace = isSpace
	}
	for start < len(s) {
		appendWord(len(s))
	}
	return words
}

// parseInlineLink parses an inline link at the start of s, and returns
// a rendered Markdown link and the total length of the raw inline link.
// If no inline link is present, it returns all zeroes.
func parseInlineLink(s string) (link string, length int) {
	if !strings.HasPrefix(s, "[[") {
		return
	}
	end := strings.Index(s, "]]")
	if end == -1 {
		return
	}
	urlEnd := strings.Index(s, "]")
	rawURL := s[2:urlEnd]
	const badURLChars = `<>"{}|\^[] ` + "`" // per RFC2396 section 2.4.3
	if strings.ContainsAny(rawURL, badURLChars) {
		return
	}
	if urlEnd == end {
		simpleURL := ""
		url, err := url.Parse(rawURL)
		if err == nil {
			// If the URL is http://foo.com, drop the http://
			// In other words, render [[http://golang.org]] as:
			//   <a href="http://golang.org">golang.org</a>
			if strings.HasPrefix(rawURL, url.Scheme+"://") {
				simpleURL = strings.TrimPrefix(rawURL, url.Scheme+"://")
			} else if strings.HasPrefix(rawURL, url.Scheme+":") {
				simpleURL = strings.TrimPrefix(rawURL, url.Scheme+":")
			}
		}
		return renderLink(rawURL, simpleURL), end + 2
	}
	if s[urlEnd:urlEnd+2] != "][" {
		return
	}
	text := s[urlEnd+2 : end]
	return renderLink(rawURL, text), end + 2
}

func renderLink(href, text string) string {
	text = font(text, false)
	if text == "" {
		text = markdownEscape(href, false)
	}
	return "[" + text + "](" + href + ")"
}
