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
	"io/ioutil"
	"log"
	"net/url"
	"regexp"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
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
	Time       time.Time
	Authors    []Author
	TitleNotes []string
	Sections   []Section
	Tags       []string
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
}

func (t Text) TemplateName() string { return "text" }

// List represents a bulleted list.
type List struct {
	Bullet []string
}

func (l List) TemplateName() string { return "list" }

// Lines is a helper for parsing line-based input.
type Lines struct {
	line int // 0 indexed, so has 1-indexed number of last line returned
	text []string
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
	return &Lines{0, lines}, nil
}

func (l *Lines) next() (text string, ok bool) {
	for {
		current := l.line
		l.line++
		if current >= len(l.text) {
			return "", false
		}
		text = l.text[current]
		// Lines starting with # are comments.
		if len(text) == 0 || text[0] != '#' {
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

	for i := lines.line; i < len(lines.text); i++ {
		if strings.HasPrefix(lines.text[i], "*") {
			break
		}

		if isSpeakerNote(lines.text[i]) {
			doc.TitleNotes = append(doc.TitleNotes, lines.text[i][2:])
		}
	}

	err = parseHeader(doc, lines)
	if err != nil {
		return nil, err
	}
	if mode&TitlesOnly != 0 {
		return doc, nil
	}

	// Authors
	if doc.Authors, err = parseAuthors(lines); err != nil {
		return nil, err
	}
	// Sections
	if doc.Sections, err = parseSections(ctx, name, lines, []int{}); err != nil {
		return nil, err
	}
	return doc, nil
}

// Parse parses a document from r. Parse reads assets used by the presentation
// from the file system using ioutil.ReadFile.
func Parse(r io.Reader, name string, mode ParseMode) (*Doc, error) {
	ctx := Context{ReadFile: ioutil.ReadFile}
	return ctx.Parse(r, name, mode)
}

// isHeading matches any section heading.
var isHeading = regexp.MustCompile(`^\*+ `)

// lesserHeading returns true if text is a heading of a lesser or equal level
// than that denoted by prefix.
func lesserHeading(text, prefix string) bool {
	return isHeading.MatchString(text) && !strings.HasPrefix(text, prefix+"*")
}

// parseSections parses Sections from lines for the section level indicated by
// number (a nil number indicates the top level).
func parseSections(ctx *Context, name string, lines *Lines, number []int) ([]Section, error) {
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
		prefix := strings.Repeat("*", len(number)+1)
		if !strings.HasPrefix(text, prefix+" ") {
			lines.back()
			break
		}
		section := Section{
			Number: append(append([]int{}, number...), i),
			Title:  text[len(prefix)+1:],
		}
		text, ok = lines.nextNonEmpty()
		for ok && !lesserHeading(text, prefix) {
			var e Elem
			r, _ := utf8.DecodeRuneInString(text)
			switch {
			case unicode.IsSpace(r):
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
				pre = strings.Replace(pre, "\t", "    ", -1) // browsers treat tabs badly
				pre = strings.TrimRightFunc(pre, unicode.IsSpace)
				e = Text{Lines: []string{pre}, Pre: true}
			case strings.HasPrefix(text, "- "):
				var b []string
				for ok && strings.HasPrefix(text, "- ") {
					b = append(b, text[2:])
					text, ok = lines.next()
				}
				lines.back()
				e = List{Bullet: b}
			case isSpeakerNote(text):
				section.Notes = append(section.Notes, text[2:])
			case strings.HasPrefix(text, prefix+"* "):
				lines.back()
				subsecs, err := parseSections(ctx, name, lines, section.Number)
				if err != nil {
					return nil, err
				}
				for _, ss := range subsecs {
					section.Elem = append(section.Elem, ss)
				}
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
			default:
				var l []string
				for ok && strings.TrimSpace(text) != "" {
					if text[0] == '.' { // Command breaks text block.
						lines.back()
						break
					}
					if strings.HasPrefix(text, `\.`) { // Backslash escapes initial period.
						text = text[1:]
					}
					l = append(l, text)
					text, ok = lines.next()
				}
				if len(l) > 0 {
					e = Text{Lines: l}
				}
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
	return sections, nil
}

func parseHeader(doc *Doc, lines *Lines) error {
	var ok bool
	// First non-empty line starts header.
	doc.Title, ok = lines.nextNonEmpty()
	if !ok {
		return errors.New("unexpected EOF; expected title")
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
		const tagPrefix = "Tags:"
		if strings.HasPrefix(text, tagPrefix) {
			tags := strings.Split(text[len(tagPrefix):], ",")
			for i := range tags {
				tags[i] = strings.TrimSpace(tags[i])
			}
			doc.Tags = append(doc.Tags, tags...)
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

func parseAuthors(lines *Lines) (authors []Author, err error) {
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
		if strings.HasPrefix(text, "* ") {
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
			el = parseURL("http://twitter.com/" + text[1:])
		case strings.Contains(text, ":"):
			el = parseURL(text)
		case strings.Contains(text, "@"):
			el = parseURL("mailto:" + text)
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

func parseURL(text string) Elem {
	u, err := url.Parse(text)
	if err != nil {
		log.Printf("Parse(%q): %v", text, err)
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
	return strings.HasPrefix(s, ": ")
}
