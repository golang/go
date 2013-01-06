// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// Collation table generator.
// Data read from the web.

package main

import (
	"archive/zip"
	"bufio"
	"bytes"
	"encoding/xml"
	"exp/locale/collate"
	"exp/locale/collate/build"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

var (
	root = flag.String("root",
		"http://unicode.org/Public/UCA/"+unicode.Version+"/CollationAuxiliary.zip",
		`URL of the Default Unicode Collation Element Table (DUCET). This can be a zip
file containing the file allkeys_CLDR.txt or an allkeys.txt file.`)
	cldr = flag.String("cldr",
		"http://www.unicode.org/Public/cldr/22/core.zip",
		"URL of CLDR archive.")
	test = flag.Bool("test", false,
		"test existing tables; can be used to compare web data with package data.")
	localFiles = flag.Bool("local", false,
		"data files have been copied to the current directory; for debugging only.")
	short = flag.Bool("short", false, `Use "short" alternatives, when available.`)
	draft = flag.Bool("draft", false, `Use draft versions, when available.`)
	tags  = flag.String("tags", "", "build tags to be included after +build directive")
	pkg   = flag.String("package", "collate",
		"the name of the package in which the generated file is to be included")

	tables = flagStringSetAllowAll("tables", "collate", "collate,chars",
		"comma-spearated list of tables to generate.")
	exclude = flagStringSet("exclude", "zh2", "",
		"comma-separated list of languages to exclude.")
	include = flagStringSet("include", "", "",
		"comma-separated list of languages to include. Include trumps exclude.")
	types = flagStringSetAllowAll("types", "", "",
		"comma-separated list of types that should be included in addition to the standard type.")
)

// stringSet implements an ordered set based on a list.  It implements flag.Value
// to allow a set to be specified as a comma-separated list.
type stringSet struct {
	s        []string
	allowed  *stringSet
	dirty    bool // needs compaction if true
	all      bool
	allowAll bool
}

func flagStringSet(name, def, allowed, usage string) *stringSet {
	ss := &stringSet{}
	if allowed != "" {
		usage += fmt.Sprintf(" (allowed values: any of %s)", allowed)
		ss.allowed = &stringSet{}
		failOnError(ss.allowed.Set(allowed))
	}
	ss.Set(def)
	flag.Var(ss, name, usage)
	return ss
}

func flagStringSetAllowAll(name, def, allowed, usage string) *stringSet {
	ss := &stringSet{allowAll: true}
	if allowed == "" {
		flag.Var(ss, name, usage+fmt.Sprintf(` Use "all" to select all.`))
	} else {
		ss.allowed = &stringSet{}
		failOnError(ss.allowed.Set(allowed))
		flag.Var(ss, name, usage+fmt.Sprintf(` (allowed values: "all" or any of %s)`, allowed))
	}
	ss.Set(def)
	return ss
}

func (ss stringSet) Len() int {
	return len(ss.s)
}

func (ss stringSet) String() string {
	return strings.Join(ss.s, ",")
}

func (ss *stringSet) Set(s string) error {
	if ss.allowAll && s == "all" {
		ss.s = nil
		ss.all = true
		return nil
	}
	ss.s = ss.s[:0]
	for _, s := range strings.Split(s, ",") {
		if s := strings.TrimSpace(s); s != "" {
			if ss.allowed != nil && !ss.allowed.contains(s) {
				return fmt.Errorf("unsupported value %q; must be one of %s", s, ss.allowed)
			}
			ss.add(s)
		}
	}
	ss.compact()
	return nil
}

func (ss *stringSet) add(s string) {
	ss.s = append(ss.s, s)
	ss.dirty = true
}

func (ss *stringSet) values() []string {
	ss.compact()
	return ss.s
}

func (ss *stringSet) contains(s string) bool {
	if ss.all {
		return true
	}
	for _, v := range ss.s {
		if v == s {
			return true
		}
	}
	return false
}

func (ss *stringSet) compact() {
	if !ss.dirty {
		return
	}
	a := ss.s
	sort.Strings(a)
	k := 0
	for i := 1; i < len(a); i++ {
		if a[k] != a[i] {
			a[k+1] = a[i]
			k++
		}
	}
	ss.s = a[:k+1]
	ss.dirty = false
}

func skipLang(l string) bool {
	if include.Len() > 0 {
		return !include.contains(l)
	}
	return exclude.contains(l)
}

func skipAlt(a string) bool {
	if *draft && a == "proposed" {
		return false
	}
	if *short && a == "short" {
		return false
	}
	return true
}

func failOnError(e error) {
	if e != nil {
		log.Panic(e)
	}
}

// openReader opens the URL or file given by url and returns it as an io.ReadCloser
// or nil on error.
func openReader(url *string) (io.ReadCloser, error) {
	if *localFiles {
		pwd, _ := os.Getwd()
		*url = "file://" + path.Join(pwd, path.Base(*url))
	}
	t := &http.Transport{}
	t.RegisterProtocol("file", http.NewFileTransport(http.Dir("/")))
	c := &http.Client{Transport: t}
	resp, err := c.Get(*url)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf(`bad GET status for "%s": %s`, *url, resp.Status)
	}
	return resp.Body, nil
}

func openArchive(url *string) *zip.Reader {
	f, err := openReader(url)
	failOnError(err)
	buffer, err := ioutil.ReadAll(f)
	f.Close()
	failOnError(err)
	archive, err := zip.NewReader(bytes.NewReader(buffer), int64(len(buffer)))
	failOnError(err)
	return archive
}

// parseUCA parses a Default Unicode Collation Element Table of the format
// specified in http://www.unicode.org/reports/tr10/#File_Format.
// It returns the variable top.
func parseUCA(builder *build.Builder) {
	var r io.ReadCloser
	var err error
	if strings.HasSuffix(*root, ".zip") {
		for _, f := range openArchive(root).File {
			if strings.HasSuffix(f.Name, "allkeys_CLDR.txt") {
				r, err = f.Open()
			}
		}
		if r == nil {
			err = fmt.Errorf("file allkeys_CLDR.txt not found in archive %q", *root)
		}
	} else {
		r, err = openReader(root)
	}
	failOnError(err)
	defer r.Close()
	input := bufio.NewReader(r)
	colelem := regexp.MustCompile(`\[([.*])([0-9A-F.]+)\]`)
	for i := 1; err == nil; i++ {
		l, prefix, e := input.ReadLine()
		err = e
		line := string(l)
		if prefix {
			log.Fatalf("%d: buffer overflow", i)
		}
		if err != nil && err != io.EOF {
			log.Fatalf("%d: %v", i, err)
		}
		if len(line) == 0 || line[0] == '#' {
			continue
		}
		if line[0] == '@' {
			// parse properties
			switch {
			case strings.HasPrefix(line[1:], "version "):
				a := strings.Split(line[1:], " ")
				if a[1] != unicode.Version {
					log.Fatalf("incompatible version %s; want %s", a[1], unicode.Version)
				}
			case strings.HasPrefix(line[1:], "backwards "):
				log.Fatalf("%d: unsupported option backwards", i)
			default:
				log.Printf("%d: unknown option %s", i, line[1:])
			}
		} else {
			// parse entries
			part := strings.Split(line, " ; ")
			if len(part) != 2 {
				log.Fatalf("%d: production rule without ';': %v", i, line)
			}
			lhs := []rune{}
			for _, v := range strings.Split(part[0], " ") {
				if v == "" {
					continue
				}
				lhs = append(lhs, rune(convHex(i, v)))
			}
			var n int
			var vars []int
			rhs := [][]int{}
			for i, m := range colelem.FindAllStringSubmatch(part[1], -1) {
				n += len(m[0])
				elem := []int{}
				for _, h := range strings.Split(m[2], ".") {
					elem = append(elem, convHex(i, h))
				}
				if m[1] == "*" {
					vars = append(vars, i)
				}
				rhs = append(rhs, elem)
			}
			if len(part[1]) < n+3 || part[1][n+1] != '#' {
				log.Fatalf("%d: expected comment; found %s", i, part[1][n:])
			}
			if *test {
				testInput.add(string(lhs))
			}
			failOnError(builder.Add(lhs, rhs, vars))
		}
	}
}

func convHex(line int, s string) int {
	r, e := strconv.ParseInt(s, 16, 32)
	if e != nil {
		log.Fatalf("%d: %v", line, e)
	}
	return int(r)
}

var testInput = stringSet{}

// LDML holds all collation information parsed from an LDML XML file.
// The format of these files is defined in http://unicode.org/reports/tr35/.
type LDML struct {
	XMLName   xml.Name `xml:"ldml"`
	Language  Attr     `xml:"identity>language"`
	Territory Attr     `xml:"identity>territory"`
	Chars     *struct {
		ExemplarCharacters []AttrValue `xml:"exemplarCharacters"`
		MoreInformaton     string      `xml:"moreInformation,omitempty"`
	} `xml:"characters"`
	Default    Attr        `xml:"collations>default"`
	Collations []Collation `xml:"collations>collation"`
}

type Attr struct {
	XMLName xml.Name
	Attr    string `xml:"type,attr"`
}

func (t Attr) String() string {
	return t.Attr
}

type AttrValue struct {
	Type  string `xml:"type,attr"`
	Key   string `xml:"key,attr,omitempty"`
	Draft string `xml:"draft,attr,omitempty"`
	Value string `xml:",innerxml"`
}

type Collation struct {
	Type                string    `xml:"type,attr"`
	Alt                 string    `xml:"alt,attr"`
	SuppressContraction string    `xml:"suppress_contractions,omitempty"`
	Settings            *Settings `xml:"settings"`
	Optimize            string    `xml:"optimize"`
	Rules               Rules     `xml:"rules"`
}

type Optimize struct {
	XMLName xml.Name `xml:"optimize"`
	Data    string   `xml:"chardata"`
}

type Suppression struct {
	XMLName xml.Name `xml:"suppress_contractions"`
	Data    string   `xml:"chardata"`
}

type Settings struct {
	Strength            string `xml:"strenght,attr,omitempty"`
	Backwards           string `xml:"backwards,attr,omitempty"`
	Normalization       string `xml:"normalization,attr,omitempty"`
	CaseLevel           string `xml:"caseLevel,attr,omitempty"`
	CaseFirst           string `xml:"caseFirst,attr,omitempty"`
	HiraganaQuarternary string `xml:"hiraganaQuartenary,attr,omitempty"`
	Numeric             string `xml:"numeric,attr,omitempty"`
	VariableTop         string `xml:"variableTop,attr,omitempty"`
}

type Rules struct {
	XMLName xml.Name   `xml:"rules"`
	Any     []RuleElem `xml:",any"`
}

type RuleElem struct {
	XMLName xml.Name
	Value   string     `xml:",innerxml"`
	Before  string     `xml:"before,attr"`
	Any     []RuleElem `xml:",any"` // for <x> elements
}

var charRe = regexp.MustCompile(`&#x([0-9A-F]*);`)
var tagRe = regexp.MustCompile(`<([a-z_]*)  */>`)

func (r *RuleElem) rewrite() {
	// Convert hexadecimal Unicode codepoint notation to a string.
	if m := charRe.FindAllStringSubmatch(r.Value, -1); m != nil {
		runes := []rune{}
		for _, sa := range m {
			runes = append(runes, rune(convHex(-1, sa[1])))
		}
		r.Value = string(runes)
	}
	// Strip spaces from reset positions.
	if m := tagRe.FindStringSubmatch(r.Value); m != nil {
		r.Value = fmt.Sprintf("<%s/>", m[1])
	}
	for _, rr := range r.Any {
		rr.rewrite()
	}
}

func decodeXML(f *zip.File) *LDML {
	r, err := f.Open()
	failOnError(err)
	d := xml.NewDecoder(r)
	var x LDML
	err = d.Decode(&x)
	failOnError(err)
	return &x
}

var mainLocales = []string{}

// charsets holds a list of exemplar characters per category.
type charSets map[string][]string

func (p charSets) fprint(w io.Writer) {
	fmt.Fprintln(w, "[exN]string{")
	for i, k := range []string{"", "contractions", "punctuation", "auxiliary", "currencySymbol", "index"} {
		if set := p[k]; len(set) != 0 {
			fmt.Fprintf(w, "\t\t%d: %q,\n", i, strings.Join(set, " "))
		}
	}
	fmt.Fprintln(w, "\t},")
}

var localeChars = make(map[string]charSets)

const exemplarHeader = `
type exemplarType int
const (
	exCharacters exemplarType = iota
	exContractions
	exPunctuation
	exAuxiliary
	exCurrency
	exIndex
	exN
)
`

func printExemplarCharacters(w io.Writer) {
	fmt.Fprintln(w, exemplarHeader)
	fmt.Fprintln(w, "var exemplarCharacters = map[string][exN]string{")
	for _, loc := range mainLocales {
		fmt.Fprintf(w, "\t%q: ", loc)
		localeChars[loc].fprint(w)
	}
	fmt.Fprintln(w, "}")
}

var mainRe = regexp.MustCompile(`.*/main/(.*)\.xml`)

// parseMain parses XML files in the main directory of the CLDR core.zip file.
func parseMain() {
	for _, f := range openArchive(cldr).File {
		if m := mainRe.FindStringSubmatch(f.Name); m != nil {
			locale := m[1]
			x := decodeXML(f)
			if skipLang(x.Language.Attr) {
				continue
			}
			if x.Chars != nil {
				for _, ec := range x.Chars.ExemplarCharacters {
					if ec.Draft != "" {
						continue
					}
					if _, ok := localeChars[locale]; !ok {
						mainLocales = append(mainLocales, locale)
						localeChars[locale] = make(charSets)
					}
					localeChars[locale][ec.Type] = parseCharacters(ec.Value)
				}
			}
		}
	}
}

func parseCharacters(chars string) []string {
	parseSingle := func(s string) (r rune, tail string, escaped bool) {
		if s[0] == '\\' {
			if s[1] == 'u' || s[1] == 'U' {
				r, _, tail, err := strconv.UnquoteChar(s, 0)
				failOnError(err)
				return r, tail, false
			} else if strings.HasPrefix(s[1:], "&amp;") {
				return '&', s[6:], false
			}
			return rune(s[1]), s[2:], true
		} else if strings.HasPrefix(s, "&quot;") {
			return '"', s[6:], false
		}
		r, sz := utf8.DecodeRuneInString(s)
		return r, s[sz:], false
	}
	chars = strings.Trim(chars, "[ ]")
	list := []string{}
	var r, last, end rune
	for len(chars) > 0 {
		if chars[0] == '{' { // character sequence
			buf := []rune{}
			for chars = chars[1:]; len(chars) > 0; {
				r, chars, _ = parseSingle(chars)
				if r == '}' {
					break
				}
				if r == ' ' {
					log.Fatalf("space not supported in sequence %q", chars)
				}
				buf = append(buf, r)
			}
			list = append(list, string(buf))
			last = 0
		} else { // single character
			escaped := false
			r, chars, escaped = parseSingle(chars)
			if r != ' ' {
				if r == '-' && !escaped {
					if last == 0 {
						log.Fatal("'-' should be preceded by a character")
					}
					end, chars, _ = parseSingle(chars)
					for ; last <= end; last++ {
						list = append(list, string(last))
					}
					last = 0
				} else {
					list = append(list, string(r))
					last = r
				}
			}
		}
	}
	return list
}

var fileRe = regexp.MustCompile(`.*/collation/(.*)\.xml`)

// parseCollation parses XML files in the collation directory of the CLDR core.zip file.
func parseCollation(b *build.Builder) {
	for _, f := range openArchive(cldr).File {
		if m := fileRe.FindStringSubmatch(f.Name); m != nil {
			lang := m[1]
			x := decodeXML(f)
			if skipLang(x.Language.Attr) {
				continue
			}
			def := "standard"
			if x.Default.Attr != "" {
				def = x.Default.Attr
			}
			todo := make(map[string]Collation)
			for _, c := range x.Collations {
				if c.Type != def && !types.contains(c.Type) {
					continue
				}
				if c.Alt != "" && skipAlt(c.Alt) {
					continue
				}
				for j := range c.Rules.Any {
					c.Rules.Any[j].rewrite()
				}
				locale := lang
				if c.Type != def {
					locale += "_u_co_" + c.Type
				}
				_, exists := todo[locale]
				if c.Alt != "" || !exists {
					todo[locale] = c
				}
			}
			for _, c := range x.Collations {
				locale := lang
				if c.Type != def {
					locale += "_u_co_" + c.Type
				}
				if d, ok := todo[locale]; ok && d.Alt == c.Alt {
					insertCollation(b, locale, &c)
				}
			}
		}
	}
}

var lmap = map[byte]collate.Level{
	'p': collate.Primary,
	's': collate.Secondary,
	't': collate.Tertiary,
	'i': collate.Identity,
}

// cldrIndex is a Unicode-reserved sentinel value used.
// We ignore any rule that starts with this rune.
// See http://unicode.org/reports/tr35/#Collation_Elements for details.
const cldrIndex = 0xFDD0

func insertTailoring(t *build.Tailoring, r RuleElem, context, extend string) {
	switch l := r.XMLName.Local; l {
	case "p", "s", "t", "i":
		if []rune(r.Value)[0] != cldrIndex {
			str := context + r.Value
			if *test {
				testInput.add(str)
			}
			err := t.Insert(lmap[l[0]], str, context+extend)
			failOnError(err)
		}
	case "pc", "sc", "tc", "ic":
		level := lmap[l[0]]
		for _, s := range r.Value {
			str := context + string(s)
			if *test {
				testInput.add(str)
			}
			err := t.Insert(level, str, context+extend)
			failOnError(err)
		}
	default:
		log.Fatalf("unsupported tag: %q", l)
	}
}

func insertCollation(builder *build.Builder, locale string, c *Collation) {
	t := builder.Tailoring(locale)
	for _, r := range c.Rules.Any {
		switch r.XMLName.Local {
		case "reset":
			if r.Before == "" {
				failOnError(t.SetAnchor(r.Value))
			} else {
				failOnError(t.SetAnchorBefore(r.Value))
			}
		case "x":
			var context, extend string
			for _, r1 := range r.Any {
				switch r1.XMLName.Local {
				case "context":
					context = r1.Value
				case "extend":
					extend = r1.Value
				}
			}
			for _, r1 := range r.Any {
				if t := r1.XMLName.Local; t == "context" || t == "extend" {
					continue
				}
				insertTailoring(t, r1, context, extend)
			}
		default:
			insertTailoring(t, r, "", "")
		}
	}
}

func testCollator(c *collate.Collator) {
	c0 := collate.New("")

	// iterator over all characters for all locales and check
	// whether Key is equal.
	buf := collate.Buffer{}

	// Add all common and not too uncommon runes to the test set.
	for i := rune(0); i < 0x30000; i++ {
		testInput.add(string(i))
	}
	for i := rune(0xE0000); i < 0xF0000; i++ {
		testInput.add(string(i))
	}
	for _, str := range testInput.values() {
		k0 := c0.KeyFromString(&buf, str)
		k := c.KeyFromString(&buf, str)
		if !bytes.Equal(k0, k) {
			failOnError(fmt.Errorf("test:%U: keys differ (%x vs %x)", []rune(str), k0, k))
		}
		buf.Reset()
	}
	fmt.Println("PASS")
}

func main() {
	flag.Parse()
	b := build.NewBuilder()
	if *root != "" {
		parseUCA(b)
	}
	if *cldr != "" {
		if tables.contains("chars") {
			parseMain()
		}
		parseCollation(b)
	}

	c, err := b.Build()
	failOnError(err)

	if *test {
		testCollator(c)
	} else {
		fmt.Println("// Generated by running")
		fmt.Printf("//  maketables -root=%s -cldr=%s\n", *root, *cldr)
		fmt.Println("// DO NOT EDIT")
		fmt.Println("// TODO: implement more compact representation for sparse blocks.")
		if *tags != "" {
			fmt.Printf("// +build %s\n", *tags)
		}
		fmt.Println("")
		fmt.Printf("package %s\n", *pkg)
		if tables.contains("collate") {
			fmt.Println("")
			_, err = b.Print(os.Stdout)
			failOnError(err)
		}
		if tables.contains("chars") {
			printExemplarCharacters(os.Stdout)
		}
	}
}
