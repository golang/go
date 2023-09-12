// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package present

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"html/template"
	"io"
	"log"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/ast"
	"github.com/yuin/goldmark/renderer/html"
	"github.com/yuin/goldmark/text"
)

var (
	parsers = make(map[string]ParseFunc)
	funcs   = template.FuncMap{}
)

// Template returns an empty template with the action functions in its FuncMap.
func Template() *template.Template {
	return template.New("").Funcs(funcs)
}

// Render renders the doc to the given writer using the provided template.
func (d *Doc) Render(w io.Writer, t *template.Template) error {
	data := struct {
		*Doc
		Template     *template.Template
		PlayEnabled  bool
		NotesEnabled bool
	}{d, t, PlayEnabled, NotesEnabled}
	return t.ExecuteTemplate(w, "root", data)
}

// Render renders the section to the given writer using the provided template.
func (s *Section) Render(w io.Writer, t *template.Template) error {
	data := struct {
		*Section
		Template    *template.Template
		PlayEnabled bool
	}{s, t, PlayEnabled}
	return t.ExecuteTemplate(w, "section", data)
}

type ParseFunc func(ctx *Context, fileName string, lineNumber int, inputLine string) (Elem, error)

// Register binds the named action, which does not begin with a period, to the
// specified parser to be invoked when the name, with a period, appears in the
// present input text.
func Register(name string, parser ParseFunc) {
	if len(name) == 0 || name[0] == ';' {
		panic("bad name in Register: " + name)
	}
	parsers["."+name] = parser
}

// Doc represents an entire document.
type Doc struct {
	Title      string
	Subtitle   string
	Summary    string
	Time       time.Time
	Authors    []Author
	TitleNotes []string
	Sections   []Section
	Tags       []string
	OldURL     []string
}

// Author represents the person who wrote and/or is presenting the document.
type Author struct {
	Elem []Elem
}

// TextElem returns the first text elements of the author details.
// This is used to display the author' name, job title, and company
// without the contact details.
func (p *Author) TextElem() (elems []Elem) {
	for _, el := range p.Elem {
		if _, ok := el.(Text); !ok {
			break
		}
		elems = append(elems, el)
	}
	return
}

// Section represents a section of a document (such as a presentation slide)
// comprising a title and a list of elements.
type Section struct {
	Number  []int
	Title   string
	ID      string // HTML anchor ID
	Elem    []Elem
	Notes   []string
	Classes []string
	Styles  []string
}

// HTMLAttributes for the section
func (s Section) HTMLAttributes() template.HTMLAttr {
	if len(s.Classes) == 0 && len(s.Styles) == 0 {
		return ""
	}

	var class string
	if len(s.Classes) > 0 {
		class = fmt.Sprintf(`class=%q`, strings.Join(s.Classes, " "))
	}
	var style string
	if len(s.Styles) > 0 {
		style = fmt.Sprintf(`style=%q`, strings.Join(s.Styles, " "))
	}
	return template.HTMLAttr(strings.Join([]string{class, style}, " "))
}

// Sections contained within the section.
func (s Section) Sections() (sections []Section) {
	for _, e := range s.Elem {
		if section, ok := e.(Section); ok {
			sections = append(sections, section)
		}
	}
	return
}

// Level returns the level of the given section.
// The document title is level 1, main section 2, etc.
func (s Section) Level() int {
	return len(s.Number) + 1
}

// FormattedNumber returns a string containing the concatenation of the
// numbers identifying a Section.
func (s Section) FormattedNumber() string {
	b := &bytes.Buffer{}
	for _, n := range s.Number {
		fmt.Fprintf(b, "%v.", n)
	}
	return b.String()
}

func (s Section) TemplateName() string { return "section" }

// Elem defines the interface for a present element. That is, something that
// can provide the name of the template used to render the element.
type Elem interface {
	TemplateName() string
}

// renderElem implements the elem template function, used to render
// sub-templates.
func renderElem(t *template.Template, e Elem) (template.HTML, error) {
	var data interface{} = e
	if s, ok := e.(Section); ok {
		data = struct {
			Section
			Template *template.Template
		}{s, t}
	}
	return execTemplate(t, e.TemplateName(), data)
}

// pageNum derives a page number from a section.
func pageNum(s Section, offset int) int {
	if len(s.Number) == 0 {
		return offset
	}
	return s.Number[0] + offset
}

func init() {
	funcs["elem"] = renderElem
	funcs["pagenum"] = pageNum
}

// execTemplate is a helper to execute a template and return the output as a
// template.HTML value.
func execTemplate(t *template.Template, name string, data interface{}) (template.HTML, error) {
	b := new(bytes.Buffer)
	err := t.ExecuteTemplate(b, name, data)
	if err != nil {
		return "", err
	}
	return template.HTML(b.String()), nil
}

// Text represents an optionally preformatted paragraph.
type Text struct {
	Lines []string
	Pre   bool
	Raw   string // original text, for Pre==true
}

func (t Text) TemplateName() string { return "text" }

// List represents a bulleted list.
type List struct {
	Bullet []string
}

func (l List) TemplateName() string { return "list" }

// Lines is a helper for parsing line-based input.
type Lines struct {
	line    int // 0 indexed, so has 1-indexed number of last line returned
	text    []string
	comment string
}

func readLines(r io.Reader) (*Lines, error) {
	var lines []string
	s := bufio.NewScanner(r)
	for s.Scan() {
		lines = append(lines, s.Text())
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	return &Lines{0, lines, "#"}, nil
}

func (l *Lines) next() (text string, ok bool) {
	for {
		current := l.line
		l.line++
		if current >= len(l.text) {
			return "", false
		}
		text = l.text[current]
		// Lines starting with l.comment are comments.
		if l.comment == "" || !strings.HasPrefix(text, l.comment) {
			ok = true
			break
		}
	}
	return
}

func (l *Lines) back() {
	l.line--
}

func (l *Lines) nextNonEmpty() (text string, ok bool) {
	for {
		text, ok = l.next()
		if !ok {
			return
		}
		if len(text) > 0 {
			break
		}
	}
	return
}

// A Context specifies the supporting context for parsing a presentation.
type Context struct {
	// ReadFile reads the file named by filename and returns the contents.
	ReadFile func(filename string) ([]byte, error)
}

// ParseMode represents flags for the Parse function.
type ParseMode int

const (
	// If set, parse only the title and subtitle.
	TitlesOnly ParseMode = 1
)

// Parse parses a document from r.
func (ctx *Context) Parse(r io.Reader, name string, mode ParseMode) (*Doc, error) {
	doc := new(Doc)
	lines, err := readLines(r)
	if err != nil {
		return nil, err
	}

	// Detect Markdown-enabled vs legacy present file.
	// Markdown-enabled files have a title line beginning with "# "
	// (like preprocessed C files of yore).
	isMarkdown := false
	for i := lines.line; i < len(lines.text); i++ {
		line := lines.text[i]
		if line == "" {
			continue
		}
		isMarkdown = strings.HasPrefix(line, "# ")
		break
	}

	sectionPrefix := "*"
	if isMarkdown {
		sectionPrefix = "##"
		lines.comment = "//"
	}

	for i := lines.line; i < len(lines.text); i++ {
		if strings.HasPrefix(lines.text[i], sectionPrefix) {
			break
		}

		if isSpeakerNote(lines.text[i]) {
			doc.TitleNotes = append(doc.TitleNotes, trimSpeakerNote(lines.text[i]))
		}
	}

	err = parseHeader(doc, isMarkdown, lines)
	if err != nil {
		return nil, err
	}
	if mode&TitlesOnly != 0 {
		return doc, nil
	}

	// Authors
	if doc.Authors, err = parseAuthors(name, sectionPrefix, lines); err != nil {
		return nil, err
	}

	// Sections
	if doc.Sections, err = parseSections(ctx, name, sectionPrefix, lines, []int{}); err != nil {
		return nil, err
	}

	return doc, nil
}

// Parse parses a document from r. Parse reads assets used by the presentation
// from the file system using os.ReadFile.
func Parse(r io.Reader, name string, mode ParseMode) (*Doc, error) {
	ctx := Context{ReadFile: os.ReadFile}
	return ctx.Parse(r, name, mode)
}

// isHeading matches any section heading.
var (
	isHeadingLegacy   = regexp.MustCompile(`^\*+( |$)`)
	isHeadingMarkdown = regexp.MustCompile(`^\#+( |$)`)
)

// lesserHeading returns true if text is a heading of a lesser or equal level
// than that denoted by prefix.
func lesserHeading(isHeading *regexp.Regexp, text, prefix string) bool {
	return isHeading.MatchString(text) && !strings.HasPrefix(text, prefix+prefix[:1])
}

// parseSections parses Sections from lines for the section level indicated by
// number (a nil number indicates the top level).
func parseSections(ctx *Context, name, prefix string, lines *Lines, number []int) ([]Section, error) {
	isMarkdown := prefix[0] == '#'
	isHeading := isHeadingLegacy
	if isMarkdown {
		isHeading = isHeadingMarkdown
	}
	var sections []Section
	for i := 1; ; i++ {
		// Next non-empty line is title.
		text, ok := lines.nextNonEmpty()
		for ok && text == "" {
			text, ok = lines.next()
		}
		if !ok {
			break
		}
		if text != prefix && !strings.HasPrefix(text, prefix+" ") {
			lines.back()
			break
		}
		// Markdown sections can end in {#id} to set the HTML anchor for the section.
		// This is nicer than the default #TOC_1_2-style anchor.
		title := strings.TrimSpace(text[len(prefix):])
		id := ""
		if isMarkdown && strings.HasSuffix(title, "}") {
			j := strings.LastIndex(title, "{#")
			if j >= 0 {
				id = title[j+2 : len(title)-1]
				title = strings.TrimSpace(title[:j])
			}
		}
		section := Section{
			Number: append(append([]int{}, number...), i),
			Title:  title,
			ID:     id,
		}
		text, ok = lines.nextNonEmpty()
		for ok && !lesserHeading(isHeading, text, prefix) {
			var e Elem
			r, _ := utf8.DecodeRuneInString(text)
			switch {
			case !isMarkdown && unicode.IsSpace(r):
				i := strings.IndexFunc(text, func(r rune) bool {
					return !unicode.IsSpace(r)
				})
				if i < 0 {
					break
				}
				indent := text[:i]
				var s []string
				for ok && (strings.HasPrefix(text, indent) || text == "") {
					if text != "" {
						text = text[i:]
					}
					s = append(s, text)
					text, ok = lines.next()
				}
				lines.back()
				pre := strings.Join(s, "\n")
				raw := pre
				pre = strings.Replace(pre, "\t", "    ", -1) // browsers treat tabs badly
				pre = strings.TrimRightFunc(pre, unicode.IsSpace)
				e = Text{Lines: []string{pre}, Pre: true, Raw: raw}
			case !isMarkdown && strings.HasPrefix(text, "- "):
				var b []string
				for {
					if strings.HasPrefix(text, "- ") {
						b = append(b, text[2:])
					} else if len(b) > 0 && strings.HasPrefix(text, " ") {
						b[len(b)-1] += "\n" + strings.TrimSpace(text)
					} else {
						break
					}
					if text, ok = lines.next(); !ok {
						break
					}
				}
				lines.back()
				e = List{Bullet: b}
			case isSpeakerNote(text):
				section.Notes = append(section.Notes, trimSpeakerNote(text))
			case strings.HasPrefix(text, prefix+prefix[:1]+" ") || text == prefix+prefix[:1]:
				lines.back()
				subsecs, err := parseSections(ctx, name, prefix+prefix[:1], lines, section.Number)
				if err != nil {
					return nil, err
				}
				for _, ss := range subsecs {
					section.Elem = append(section.Elem, ss)
				}
			case strings.HasPrefix(text, prefix+prefix[:1]):
				return nil, fmt.Errorf("%s:%d: badly nested section inside %s: %s", name, lines.line, prefix, text)
			case strings.HasPrefix(text, "."):
				args := strings.Fields(text)
				if args[0] == ".background" {
					section.Classes = append(section.Classes, "background")
					section.Styles = append(section.Styles, "background-image: url('"+args[1]+"')")
					break
				}
				parser := parsers[args[0]]
				if parser == nil {
					return nil, fmt.Errorf("%s:%d: unknown command %q", name, lines.line, text)
				}
				t, err := parser(ctx, name, lines.line, text)
				if err != nil {
					return nil, err
				}
				e = t

			case isMarkdown:
				// Collect Markdown lines, including blank lines and indented text.
				var block []string
				endLine, endBlock := lines.line-1, -1 // end is last non-empty line
				for ok {
					trim := strings.TrimSpace(text)
					if trim != "" {
						// Command breaks text block.
						// Section heading breaks text block in markdown.
						if text[0] == '.' || text[0] == '#' || isSpeakerNote(text) {
							break
						}
						if strings.HasPrefix(text, `\.`) { // Backslash escapes initial period.
							text = text[1:]
						}
						endLine, endBlock = lines.line, len(block)
					}
					block = append(block, text)
					text, ok = lines.next()
				}
				block = block[:endBlock+1]
				lines.line = endLine + 1
				if len(block) == 0 {
					break
				}

				// Replace all leading tabs with 4 spaces,
				// which render better in code blocks.
				// CommonMark defines that for parsing the structure of the file
				// a tab is equivalent to 4 spaces, so this change won't
				// affect the later parsing at all.
				// An alternative would be to apply this to code blocks after parsing,
				// at the same time that we update <a> targets, but that turns out
				// to be quite difficult to modify in the AST.
				for i, line := range block {
					if len(line) > 0 && line[0] == '\t' {
						short := strings.TrimLeft(line, "\t")
						line = strings.Repeat("    ", len(line)-len(short)) + short
						block[i] = line
					}
				}
				html, err := renderMarkdown([]byte(strings.Join(block, "\n")))
				if err != nil {
					return nil, err
				}
				e = HTML{HTML: html}

			default:
				// Collect text lines.
				var block []string
				for ok && strings.TrimSpace(text) != "" {
					// Command breaks text block.
					// Section heading breaks text block in markdown.
					if text[0] == '.' || isSpeakerNote(text) {
						lines.back()
						break
					}
					if strings.HasPrefix(text, `\.`) { // Backslash escapes initial period.
						text = text[1:]
					}
					block = append(block, text)
					text, ok = lines.next()
				}
				if len(block) == 0 {
					break
				}
				e = Text{Lines: block}
			}
			if e != nil {
				section.Elem = append(section.Elem, e)
			}
			text, ok = lines.nextNonEmpty()
		}
		if isHeading.MatchString(text) {
			lines.back()
		}
		sections = append(sections, section)
	}

	if len(sections) == 0 {
		return nil, fmt.Errorf("%s:%d: unexpected line: %s", name, lines.line+1, lines.text[lines.line])
	}
	return sections, nil
}

func parseHeader(doc *Doc, isMarkdown bool, lines *Lines) error {
	var ok bool
	// First non-empty line starts header.
	doc.Title, ok = lines.nextNonEmpty()
	if !ok {
		return errors.New("unexpected EOF; expected title")
	}
	if isMarkdown {
		doc.Title = strings.TrimSpace(strings.TrimPrefix(doc.Title, "#"))
	}

	for {
		text, ok := lines.next()
		if !ok {
			return errors.New("unexpected EOF")
		}
		if text == "" {
			break
		}
		if isSpeakerNote(text) {
			continue
		}
		if strings.HasPrefix(text, "Tags:") {
			tags := strings.Split(text[len("Tags:"):], ",")
			for i := range tags {
				tags[i] = strings.TrimSpace(tags[i])
			}
			doc.Tags = append(doc.Tags, tags...)
		} else if strings.HasPrefix(text, "Summary:") {
			doc.Summary = strings.TrimSpace(text[len("Summary:"):])
		} else if strings.HasPrefix(text, "OldURL:") {
			doc.OldURL = append(doc.OldURL, strings.TrimSpace(text[len("OldURL:"):]))
		} else if t, ok := parseTime(text); ok {
			doc.Time = t
		} else if doc.Subtitle == "" {
			doc.Subtitle = text
		} else {
			return fmt.Errorf("unexpected header line: %q", text)
		}
	}
	return nil
}

func parseAuthors(name, sectionPrefix string, lines *Lines) (authors []Author, err error) {
	// This grammar demarcates authors with blanks.

	// Skip blank lines.
	if _, ok := lines.nextNonEmpty(); !ok {
		return nil, errors.New("unexpected EOF")
	}
	lines.back()

	var a *Author
	for {
		text, ok := lines.next()
		if !ok {
			return nil, errors.New("unexpected EOF")
		}

		// If we find a section heading, we're done.
		if strings.HasPrefix(text, sectionPrefix) {
			lines.back()
			break
		}

		if isSpeakerNote(text) {
			continue
		}

		// If we encounter a blank we're done with this author.
		if a != nil && len(text) == 0 {
			authors = append(authors, *a)
			a = nil
			continue
		}
		if a == nil {
			a = new(Author)
		}

		// Parse the line. Those that
		// - begin with @ are twitter names,
		// - contain slashes are links, or
		// - contain an @ symbol are an email address.
		// The rest is just text.
		var el Elem
		switch {
		case strings.HasPrefix(text, "@"):
			el = parseAuthorURL(name, "http://twitter.com/"+text[1:])
		case strings.Contains(text, ":"):
			el = parseAuthorURL(name, text)
		case strings.Contains(text, "@"):
			el = parseAuthorURL(name, "mailto:"+text)
		}
		if l, ok := el.(Link); ok {
			l.Label = text
			el = l
		}
		if el == nil {
			el = Text{Lines: []string{text}}
		}
		a.Elem = append(a.Elem, el)
	}
	if a != nil {
		authors = append(authors, *a)
	}
	return authors, nil
}

func parseAuthorURL(name, text string) Elem {
	u, err := url.Parse(text)
	if err != nil {
		log.Printf("parsing %s author block: invalid URL %q: %v", name, text, err)
		return nil
	}
	return Link{URL: u}
}

func parseTime(text string) (t time.Time, ok bool) {
	t, err := time.Parse("15:04 2 Jan 2006", text)
	if err == nil {
		return t, true
	}
	t, err = time.Parse("2 Jan 2006", text)
	if err == nil {
		// at 11am UTC it is the same date everywhere
		t = t.Add(time.Hour * 11)
		return t, true
	}
	return time.Time{}, false
}

func isSpeakerNote(s string) bool {
	return strings.HasPrefix(s, ": ") || s == ":"
}

func trimSpeakerNote(s string) string {
	if s == ":" {
		return ""
	}
	return strings.TrimPrefix(s, ": ")
}

func renderMarkdown(input []byte) (template.HTML, error) {
	md := goldmark.New(goldmark.WithRendererOptions(html.WithUnsafe()))
	reader := text.NewReader(input)
	doc := md.Parser().Parse(reader)
	fixupMarkdown(doc)
	var b strings.Builder
	if err := md.Renderer().Render(&b, input, doc); err != nil {
		return "", err
	}
	return template.HTML(b.String()), nil
}

func fixupMarkdown(n ast.Node) {
	ast.Walk(n, func(n ast.Node, entering bool) (ast.WalkStatus, error) {
		if entering {
			switch n := n.(type) {
			case *ast.Link:
				n.SetAttributeString("target", []byte("_blank"))
				// https://developers.google.com/web/tools/lighthouse/audits/noopener
				n.SetAttributeString("rel", []byte("noopener"))
			}
		}
		return ast.WalkContinue, nil
	})
}
