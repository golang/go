// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package markdown

import (
	"bytes"
	"fmt"
	"strings"
	"unicode/utf8"

	"golang.org/x/text/cases"
)

func parseLinkRefDef(p buildState, s string) (int, bool) {
	// “A link reference definition consists of a link label,
	// optionally preceded by up to three spaces of indentation,
	// followed by a colon (:),
	// optional spaces or tabs (including up to one line ending),
	// a link destination,
	// optional spaces or tabs (including up to one line ending),
	// and an optional link title,
	// which if it is present must be separated from the link destination
	// by spaces or tabs. No further character may occur.”
	i := skipSpace(s, 0)
	label, i, ok := parseLinkLabel(p.(*parseState), s, i)
	if !ok || i >= len(s) || s[i] != ':' {
		return 0, false
	}
	i = skipSpace(s, i+1)
	suf := s[i:]
	dest, i, ok := parseLinkDest(s, i)
	if !ok {
		if suf != "" && suf[0] == '<' {
			// Goldmark treats <<> as a link definition.
			p.(*parseState).corner = true
		}
		return 0, false
	}
	moved := false
	for i < len(s) && (s[i] == ' ' || s[i] == '\t') {
		moved = true
		i++
	}

	// Take title if present and doesn't break parse.
	j := i
	if j >= len(s) || s[j] == '\n' {
		moved = true
		if j < len(s) {
			j++
		}
	}

	var title string
	var titleChar byte
	var corner bool
	if moved {
		for j < len(s) && (s[j] == ' ' || s[j] == '\t') {
			j++
		}
		if t, c, j, ok := parseLinkTitle(s, j); ok {
			for j < len(s) && (s[j] == ' ' || s[j] == '\t') {
				j++
			}
			if j >= len(s) || s[j] == '\n' {
				i = j
				if t == "" {
					// Goldmark adds title="" in this case.
					// We do not, nor does the Dingus.
					corner = true
				}
				title = t
				titleChar = c
			}
		}
	}

	// Must end line. Already trimmed spaces.
	if i < len(s) && s[i] != '\n' {
		return 0, false
	}
	if i < len(s) {
		i++
	}

	label = normalizeLabel(label)
	if p.link(label) == nil {
		p.defineLink(label, &Link{URL: dest, Title: title, TitleChar: titleChar, corner: corner})
	}
	return i, true
}

func parseLinkTitle(s string, i int) (title string, char byte, next int, found bool) {
	if i < len(s) && (s[i] == '"' || s[i] == '\'' || s[i] == '(') {
		want := s[i]
		if want == '(' {
			want = ')'
		}
		j := i + 1
		for ; j < len(s); j++ {
			if s[j] == want {
				title := s[i+1 : j]
				// TODO: Validate title?
				return mdUnescaper.Replace(title), want, j + 1, true
			}
			if s[j] == '(' && want == ')' {
				break
			}
			if s[j] == '\\' && j+1 < len(s) {
				j++
			}
		}
	}
	return "", 0, 0, false
}

func parseLinkLabel(p *parseState, s string, i int) (string, int, bool) {
	// “A link label begins with a left bracket ([) and ends with
	// the first right bracket (]) that is not backslash-escaped.
	// Between these brackets there must be at least one character
	// that is not a space, tab, or line ending.
	// Unescaped square bracket characters are not allowed
	// inside the opening and closing square brackets of link labels.
	// A link label can have at most 999 characters inside the square brackets.”
	if i >= len(s) || s[i] != '[' {
		return "", 0, false
	}
	j := i + 1
	for ; j < len(s); j++ {
		if s[j] == ']' {
			if j-(i+1) > 999 {
				// Goldmark does not apply 999 limit.
				p.corner = true
				break
			}
			if label := trimSpaceTabNewline(s[i+1 : j]); label != "" {
				// Note: CommonMark Dingus does not escape.
				return label, j + 1, true
			}
			break
		}
		if s[j] == '[' {
			break
		}
		if s[j] == '\\' && j+1 < len(s) {
			j++
		}
	}
	return "", 0, false
}

func normalizeLabel(s string) string {
	if strings.Contains(s, "[") || strings.Contains(s, "]") {
		// Labels cannot have [ ] so avoid the work of translating.
		// This is especially important for pathlogical cases like
		// [[[[[[[[[[a]]]]]]]]]] which would otherwise generate quadratic
		// amounts of garbage.
		return ""
	}

	// “To normalize a label, strip off the opening and closing brackets,
	// perform the Unicode case fold, strip leading and trailing spaces, tabs, and line endings,
	// and collapse consecutive internal spaces, tabs, and line endings to a single space.”
	s = trimSpaceTabNewline(s)
	var b strings.Builder
	space := false
	hi := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		case ' ', '\t', '\n':
			space = true
			continue
		default:
			if space {
				b.WriteByte(' ')
				space = false
			}
			if 'A' <= c && c <= 'Z' {
				c += 'a' - 'A'
			}
			if c >= 0x80 {
				hi = true
			}
			b.WriteByte(c)
		}
	}
	s = b.String()
	if hi {
		s = cases.Fold().String(s)
	}
	return s
}

func parseLinkDest(s string, i int) (string, int, bool) {
	if i >= len(s) {
		return "", 0, false
	}

	// “A sequence of zero or more characters between an opening < and a closing >
	// that contains no line endings or unescaped < or > characters,”
	if s[i] == '<' {
		for j := i + 1; ; j++ {
			if j >= len(s) || s[j] == '\n' || s[j] == '<' {
				return "", 0, false
			}
			if s[j] == '>' {
				// TODO unescape?
				return mdUnescape(s[i+1 : j]), j + 1, true
			}
			if s[j] == '\\' {
				j++
			}
		}
	}

	// “or a nonempty sequence of characters that does not start with <,
	// does not include ASCII control characters or space character,
	// and includes parentheses only if (a) they are backslash-escaped
	// or (b) they are part of a balanced pair of unescaped parentheses.
	depth := 0
	j := i
Loop:
	for ; j < len(s); j++ {
		switch s[j] {
		case '(':
			depth++
			if depth > 32 {
				// Avoid quadratic inputs by stopping if too deep.
				// This is the same depth that cmark-gfm uses.
				return "", 0, false
			}
		case ')':
			if depth == 0 {
				break Loop
			}
			depth--
		case '\\':
			if j+1 < len(s) {
				if s[j+1] == ' ' || s[j+1] == '\t' {
					return "", 0, false
				}
				j++
			}
		case ' ', '\t', '\n':
			break Loop
		}
	}

	dest := s[i:j]
	// TODO: Validate dest?
	// TODO: Unescape?
	// NOTE: CommonMark Dingus does not reject control characters.
	return mdUnescape(dest), j, true
}

func parseAutoLinkURI(s string, i int) (Inline, int, bool) {
	// CommonMark 0.30:
	//
	//	For purposes of this spec, a scheme is any sequence of 2–32 characters
	//	beginning with an ASCII letter and followed by any combination of
	//	ASCII letters, digits, or the symbols plus (”+”), period (”.”), or
	//	hyphen (”-”).
	//
	//	An absolute URI, for these purposes, consists of a scheme followed by
	//	a colon (:) followed by zero or more characters other ASCII control
	//	characters, space, <, and >. If the URI includes these characters,
	//	they must be percent-encoded (e.g. %20 for a space).

	j := i
	if j+1 >= len(s) || s[j] != '<' || !isLetter(s[j+1]) {
		return nil, 0, false
	}
	j++
	for j < len(s) && isScheme(s[j]) && j-(i+1) <= 32 {
		j++
	}
	if j-(i+1) < 2 || j-(i+1) > 32 || j >= len(s) || s[j] != ':' {
		return nil, 0, false
	}
	j++
	for j < len(s) && isURL(s[j]) {
		j++
	}
	if j >= len(s) || s[j] != '>' {
		return nil, 0, false
	}
	link := s[i+1 : j]
	// link = mdUnescaper.Replace(link)
	return &AutoLink{link, link}, j + 1, true
}

func parseAutoLinkEmail(s string, i int) (Inline, int, bool) {
	// CommonMark 0.30:
	//
	//	An email address, for these purposes, is anything that matches
	//	the non-normative regex from the HTML5 spec:
	//
	//	/^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$/

	j := i
	if j+1 >= len(s) || s[j] != '<' || !isUser(s[j+1]) {
		return nil, 0, false
	}
	j++
	for j < len(s) && isUser(s[j]) {
		j++
	}
	if j >= len(s) || s[j] != '@' {
		return nil, 0, false
	}
	for {
		j++
		n, ok := skipDomainElem(s[j:])
		if !ok {
			return nil, 0, false
		}
		j += n
		if j >= len(s) || s[j] != '.' && s[j] != '>' {
			return nil, 0, false
		}
		if s[j] == '>' {
			break
		}
	}
	email := s[i+1 : j]
	return &AutoLink{email, "mailto:" + email}, j + 1, true
}

func isUser(c byte) bool {
	if isLetterDigit(c) {
		return true
	}
	s := ".!#$%&'*+/=?^_`{|}~-"
	for i := 0; i < len(s); i++ {
		if c == s[i] {
			return true
		}
	}
	return false
}

func isHexDigit(c byte) bool {
	return 'A' <= c && c <= 'F' || 'a' <= c && c <= 'f' || '0' <= c && c <= '9'
}

func isDigit(c byte) bool {
	return '0' <= c && c <= '9'
}

func skipDomainElem(s string) (int, bool) {
	// String of LDH, up to 63 in length, with LetterDigit
	// at both ends (1-letter/digit names are OK).
	// Aka /[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?/.
	if len(s) < 1 || !isLetterDigit(s[0]) {
		return 0, false
	}
	i := 1
	for i < len(s) && isLDH(s[i]) && i <= 63 {
		i++
	}
	if i > 63 || !isLetterDigit(s[i-1]) {
		return 0, false
	}
	return i, true
}

func isScheme(c byte) bool {
	return isLetterDigit(c) || c == '+' || c == '.' || c == '-'
}

func isURL(c byte) bool {
	return c > ' ' && c != '<' && c != '>'
}

type AutoLink struct {
	Text string
	URL  string
}

func (*AutoLink) Inline() {}

func (x *AutoLink) PrintHTML(buf *bytes.Buffer) {
	fmt.Fprintf(buf, "<a href=\"%s\">%s</a>", htmlLinkEscaper.Replace(x.URL), htmlEscaper.Replace(x.Text))
}

func (x *AutoLink) printMarkdown(buf *bytes.Buffer) {
	fmt.Fprintf(buf, "<%s>", x.Text)
}

func (x *AutoLink) PrintText(buf *bytes.Buffer) {
	fmt.Fprintf(buf, "%s", htmlEscaper.Replace(x.Text))
}

type Link struct {
	Inner     []Inline
	URL       string
	Title     string
	TitleChar byte // ', " or )
	corner    bool
}

func (*Link) Inline() {}

func (x *Link) PrintHTML(buf *bytes.Buffer) {
	fmt.Fprintf(buf, "<a href=\"%s\"", htmlLinkEscaper.Replace(x.URL))
	if x.Title != "" {
		fmt.Fprintf(buf, " title=\"%s\"", htmlQuoteEscaper.Replace(x.Title))
	}
	buf.WriteString(">")
	for _, c := range x.Inner {
		c.PrintHTML(buf)
	}
	buf.WriteString("</a>")
}

func (x *Link) printMarkdown(buf *bytes.Buffer) {
	buf.WriteByte('[')
	x.printRemainingMarkdown(buf)
}

func (x *Link) printRemainingMarkdown(buf *bytes.Buffer) {
	for _, c := range x.Inner {
		c.printMarkdown(buf)
	}
	buf.WriteString("](")
	buf.WriteString(x.URL)
	printLinkTitleMarkdown(buf, x.Title, x.TitleChar)
	buf.WriteByte(')')
}

func printLinkTitleMarkdown(buf *bytes.Buffer, title string, titleChar byte) {
	if title == "" {
		return
	}
	closeChar := titleChar
	openChar := closeChar
	if openChar == ')' {
		openChar = '('
	}
	fmt.Fprintf(buf, " %c%s%c", openChar, title /*TODO(jba): escape*/, closeChar)
}

func (x *Link) PrintText(buf *bytes.Buffer) {
	for _, c := range x.Inner {
		c.PrintText(buf)
	}
}

type Image struct {
	Inner     []Inline
	URL       string
	Title     string
	TitleChar byte
	corner    bool
}

func (*Image) Inline() {}

func (x *Image) PrintHTML(buf *bytes.Buffer) {
	fmt.Fprintf(buf, "<img src=\"%s\"", htmlLinkEscaper.Replace(x.URL))
	fmt.Fprintf(buf, " alt=\"")
	i := buf.Len()
	for _, c := range x.Inner {
		c.PrintText(buf)
	}
	// GitHub and Goldmark both rewrite \n to space
	// but the Dingus does not.
	// The spec says title can be split across lines but not
	// what happens at that point.
	out := buf.Bytes()
	for ; i < len(out); i++ {
		if out[i] == '\n' {
			out[i] = ' '
		}
	}
	fmt.Fprintf(buf, "\"")
	if x.Title != "" {
		fmt.Fprintf(buf, " title=\"%s\"", htmlQuoteEscaper.Replace(x.Title))
	}
	buf.WriteString(" />")
}

func (x *Image) printMarkdown(buf *bytes.Buffer) {
	buf.WriteString("![")
	(*Link)(x).printRemainingMarkdown(buf)
}

func (x *Image) PrintText(buf *bytes.Buffer) {
	for _, c := range x.Inner {
		c.PrintText(buf)
	}
}

// GitHub Flavored Markdown autolinks extension
// https://github.github.com/gfm/#autolinks-extension-

// autoLinkMore rewrites any extended autolinks in the body
// and returns the result.
//
// body is a list of Plain, Emph, Strong, and Del nodes.
// Two Plains only appear consecutively when one is a
// potential emphasis marker that ended up being plain after all, like "_" or "**".
// There are no Link nodes.
//
// The GitHub “spec” declares that “autolinks can only come at the
// beginning of a line, after whitespace, or any of the delimiting
// characters *, _, ~, and (”. However, the GitHub web site does not
// enforce this rule: text like "$abc@def.ghi is my email" links the
// text following the $ as an email address. It appears the actual rule
// is that autolinks cannot come after ASCII letters, although they can
// come after numbers or Unicode letters.
// Since the only point of implementing GitHub Flavored Markdown
// is to match GitHub's behavior, we do what they do, not what they say,
// at least for now.
func (p *parseState) autoLinkText(list []Inline) []Inline {
	if !p.AutoLinkText {
		return list
	}

	var out []Inline // allocated lazily when we first change list
	for i, x := range list {
		switch x := x.(type) {
		case *Plain:
			if rewrite := p.autoLinkPlain(x.Text); rewrite != nil {
				if out == nil {
					out = append(out, list[:i]...)
				}
				out = append(out, rewrite...)
				continue
			}
		case *Strong:
			x.Inner = p.autoLinkText(x.Inner)
		case *Del:
			x.Inner = p.autoLinkText(x.Inner)
		case *Emph:
			x.Inner = p.autoLinkText(x.Inner)
		}
		if out != nil {
			out = append(out, x)
		}
	}
	if out == nil {
		return list
	}
	return out
}

func (p *parseState) autoLinkPlain(s string) []Inline {
	vd := &validDomainChecker{s: s}
	var out []Inline
Restart:
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == '@' {
			if before, link, after, ok := p.parseAutoEmail(s, i); ok {
				if before != "" {
					out = append(out, &Plain{Text: before})
				}
				out = append(out, link)
				vd.skip(len(s) - len(after))
				s = after
				goto Restart
			}
		}

		if (c == 'h' || c == 'm' || c == 'x' || c == 'w') && (i == 0 || !isLetter(s[i-1])) {
			if link, after, ok := p.parseAutoProto(s, i, vd); ok {
				if i > 0 {
					out = append(out, &Plain{Text: s[:i]})
				}
				out = append(out, link)
				vd.skip(len(s) - len(after))
				s = after
				goto Restart
			}
		}
	}
	if out == nil {
		return nil
	}
	out = append(out, &Plain{Text: s})
	return out
}

func (p *parseState) parseAutoProto(s string, i int, vd *validDomainChecker) (link *Link, after string, found bool) {
	if s == "" {
		return
	}
	switch s[i] {
	case 'h':
		var n int
		if strings.HasPrefix(s[i:], "https://") {
			n = len("https://")
		} else if strings.HasPrefix(s[i:], "http://") {
			n = len("http://")
		} else {
			return
		}
		return p.parseAutoHTTP(s[i:i+n], s, i, i+n, i+n+1, vd)
	case 'w':
		if !strings.HasPrefix(s[i:], "www.") {
			return
		}
		// GitHub Flavored Markdown says to use http://,
		// but it's not 1985 anymore. We live in the https:// future
		// (unless the parser is explicitly configured otherwise).
		// People who really care in their docs can write http:// themselves.
		scheme := "https://"
		if p.AutoLinkAssumeHTTP {
			scheme = "http://"
		}
		return p.parseAutoHTTP(scheme, s, i, i, i+3, vd)
	case 'm':
		if !strings.HasPrefix(s[i:], "mailto:") {
			return
		}
		return p.parseAutoMailto(s, i)
	case 'x':
		if !strings.HasPrefix(s[i:], "xmpp:") {
			return
		}
		return p.parseAutoXmpp(s, i)
	}
	return
}

// parseAutoWWW parses an extended www autolink.
// https://github.github.com/gfm/#extended-www-autolink
func (p *parseState) parseAutoHTTP(scheme, s string, textstart, start, min int, vd *validDomainChecker) (link *Link, after string, found bool) {
	n, ok := vd.parseValidDomain(start)
	if !ok {
		return
	}
	i := start + n
	domEnd := i

	// “After a valid domain, zero or more non-space non-< characters may follow.”
	paren := 0
	for i < len(s) {
		r, n := utf8.DecodeRuneInString(s[i:])
		if isUnicodeSpace(r) || r == '<' {
			break
		}
		if r == '(' {
			paren++
		}
		if r == ')' {
			paren--
		}
		i += n
	}

	// https://github.github.com/gfm/#extended-autolink-path-validation
Trim:
	for i > min {
		switch s[i-1] {
		case '?', '!', '.', ',', ':', '@', '_', '~':
			// Trim certain trailing punctuation.
			i--
			continue Trim

		case ')':
			// Trim trailing unmatched (by count only) parens.
			if paren < 0 {
				for s[i-1] == ')' && paren < 0 {
					paren++
					i--
				}
				continue Trim
			}

		case ';':
			// Trim entity reference.
			// After doing the work of the scan, we either cut that part off the string
			// or we stop the trimming entirely, so there's no chance of repeating
			// the scan on a future iteration and going accidentally quadratic.
			// Even though the Markdown spec already requires having a complete
			// list of all the HTML entities, the GitHub definition here just requires
			// "looks like" an entity, meaning its an ampersand, letters/digits, and semicolon.
			for j := i - 2; j > start; j-- {
				if j < i-2 && s[j] == '&' {
					i = j
					continue Trim
				}
				if !isLetterDigit(s[j]) {
					break Trim
				}
			}
		}
		break Trim
	}

	// According to the literal text of the GitHub Flavored Markdown spec
	// and the actual behavior on GitHub,
	// www.example.com$foo turns into <a href="https://www.example.com$foo">,
	// but that makes the character restrictions in the valid-domain check
	// almost meaningless. So we insist that when all is said and done,
	// if the domain is followed by anything, that thing must be a slash,
	// even though GitHub is not that picky.
	// People might complain about www.example.com:1234 not working,
	// but if you want to get fancy with that kind of thing, just write http:// in front.
	if textstart == start && i > domEnd && s[domEnd] != '/' {
		i = domEnd
	}

	if i < min {
		return
	}

	link = &Link{
		Inner: []Inline{&Plain{Text: s[textstart:i]}},
		URL:   scheme + s[start:i],
	}
	return link, s[i:], true
}

type validDomainChecker struct {
	s   string
	cut int // before this index, no valid domains
}

func (v *validDomainChecker) skip(i int) {
	v.s = v.s[i:]
	v.cut -= i
}

// parseValidDomain parses a valid domain.
// https://github.github.com/gfm/#valid-domain
//
// If s starts with a valid domain, parseValidDomain returns
// the length of that domain and true. If s does not start with
// a valid domain, parseValidDomain returns n, false,
// where n is the length of a prefix guaranteed not to be acceptable
// to any future call to parseValidDomain.
//
// “A valid domain consists of segments of alphanumeric characters,
// underscores (_) and hyphens (-) separated by periods (.).
// There must be at least one period, and no underscores may be
// present in the last two segments of the domain.”
//
// The spec does not spell out whether segments can be empty.
// Empirically, in GitHub's implementation they can.
func (v *validDomainChecker) parseValidDomain(start int) (n int, found bool) {
	if start < v.cut {
		return 0, false
	}
	i := start
	dots := 0
	for ; i < len(v.s); i++ {
		c := v.s[i]
		if c == '_' {
			dots = -2
			continue
		}
		if c == '.' {
			dots++
			continue
		}
		if !isLDH(c) {
			break
		}
	}
	if dots >= 0 && i > start {
		return i - start, true
	}
	v.cut = i
	return 0, false
}

func (p *parseState) parseAutoEmail(s string, i int) (before string, link *Link, after string, ok bool) {
	if s[i] != '@' {
		return
	}

	// “One ore more characters which are alphanumeric, or ., -, _, or +.”
	j := i
	for j > 0 && (isLDH(s[j-1]) || s[j-1] == '_' || s[j-1] == '+' || s[j-1] == '.') {
		j--
	}
	if i-j < 1 {
		return
	}

	// “One or more characters which are alphanumeric, or - or _, separated by periods (.).
	// There must be at least one period. The last character must not be one of - or _.”
	dots := 0
	k := i + 1
	for k < len(s) && (isLDH(s[k]) || s[k] == '_' || s[k] == '.') {
		if s[k] == '.' {
			if s[k-1] == '.' {
				// Empirically, .. stops the scan but foo@.bar is fine.
				break
			}
			dots++
		}
		k++
	}

	// “., -, and _ can occur on both sides of the @, but only . may occur at the end
	// of the email address, in which case it will not be considered part of the address”
	if s[k-1] == '.' {
		dots--
		k--
	}
	if s[k-1] == '-' || s[k-1] == '_' {
		return
	}
	if k-(i+1)-dots < 2 || dots < 1 {
		return
	}

	link = &Link{
		Inner: []Inline{&Plain{Text: s[j:k]}},
		URL:   "mailto:" + s[j:k],
	}
	return s[:j], link, s[k:], true
}

func (p *parseState) parseAutoMailto(s string, i int) (link *Link, after string, ok bool) {
	j := i + len("mailto:")
	for j < len(s) && (isLDH(s[j]) || s[j] == '_' || s[j] == '+' || s[j] == '.') {
		j++
	}
	if j >= len(s) || s[j] != '@' {
		return
	}
	before, link, after, ok := p.parseAutoEmail(s[i:], j-i)
	if before != "mailto:" || !ok {
		return nil, "", false
	}
	link.Inner[0] = &Plain{Text: s[i : len(s)-len(after)]}
	return link, after, true
}

func (p *parseState) parseAutoXmpp(s string, i int) (link *Link, after string, ok bool) {
	j := i + len("xmpp:")
	for j < len(s) && (isLDH(s[j]) || s[j] == '_' || s[j] == '+' || s[j] == '.') {
		j++
	}
	if j >= len(s) || s[j] != '@' {
		return
	}
	before, link, after, ok := p.parseAutoEmail(s[i:], j-i)
	if before != "xmpp:" || !ok {
		return nil, "", false
	}
	if after != "" && after[0] == '/' {
		k := 1
		for k < len(after) && (isLetterDigit(after[k]) || after[k] == '@' || after[k] == '.') {
			k++
		}
		after = after[k:]
	}
	url := s[i : len(s)-len(after)]
	link.Inner[0] = &Plain{Text: url}
	link.URL = url
	return link, after, true
}
