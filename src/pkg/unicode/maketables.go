// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Unicode table generator.
// Data read from the web.

package main

import (
	"bufio"
	"flag"
	"fmt"
	"http"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"regexp"
	"unicode"
)

func main() {
	flag.Parse()
	loadChars() // always needed
	printCategories()
	printScriptOrProperty(false)
	printScriptOrProperty(true)
	printCases()
}

var dataURL = flag.String("data", "", "full URL for UnicodeData.txt; defaults to --url/UnicodeData.txt")
var url = flag.String("url",
	"http://www.unicode.org/Public/5.2.0/ucd/",
	"URL of Unicode database directory")
var tablelist = flag.String("tables",
	"all",
	"comma-separated list of which tables to generate; can be letter")
var scriptlist = flag.String("scripts",
	"all",
	"comma-separated list of which script tables to generate")
var proplist = flag.String("props",
	"all",
	"comma-separated list of which property tables to generate")
var cases = flag.Bool("cases",
	true,
	"generate case tables")
var test = flag.Bool("test",
	false,
	"test existing tables; can be used to compare web data with package data")

var scriptRe = regexp.MustCompile(`([0-9A-F]+)(\.\.[0-9A-F]+)? *; ([A-Za-z_]+)`)
var die = log.New(os.Stderr, nil, "", log.Lexit|log.Lshortfile)

var category = map[string]bool{"letter": true} // Nd Lu etc. letter is a special case

// UnicodeData.txt has form:
//	0037;DIGIT SEVEN;Nd;0;EN;;7;7;7;N;;;;;
//	007A;LATIN SMALL LETTER Z;Ll;0;L;;;;;N;;;005A;;005A
// See http://www.unicode.org/Public/5.1.0/ucd/UCD.html for full explanation
// The fields:
const (
	FCodePoint = iota
	FName
	FGeneralCategory
	FCanonicalCombiningClass
	FBidiClass
	FDecompositionType
	FDecompositionMapping
	FNumericType
	FNumericValue
	FBidiMirrored
	FUnicode1Name
	FISOComment
	FSimpleUppercaseMapping
	FSimpleLowercaseMapping
	FSimpleTitlecaseMapping
	NumField

	MaxChar = 0x10FFFF // anything above this shouldn't exist
)

var fieldName = []string{
	"CodePoint",
	"Name",
	"GeneralCategory",
	"CanonicalCombiningClass",
	"BidiClass",
	"DecompositionType",
	"DecompositionMapping",
	"NumericType",
	"NumericValue",
	"BidiMirrored",
	"Unicode1Name",
	"ISOComment",
	"SimpleUppercaseMapping",
	"SimpleLowercaseMapping",
	"SimpleTitlecaseMapping",
}

// This contains only the properties we're interested in.
type Char struct {
	field     []string // debugging only; could be deleted if we take out char.dump()
	codePoint uint32   // if zero, this index is not a valid code point.
	category  string
	upperCase int
	lowerCase int
	titleCase int
}

// Scripts.txt has form:
//	A673          ; Cyrillic # Po       SLAVONIC ASTERISK
//	A67C..A67D    ; Cyrillic # Mn   [2] COMBINING CYRILLIC KAVYKA..COMBINING CYRILLIC PAYEROK
// See http://www.unicode.org/Public/5.1.0/ucd/UCD.html for full explanation

type Script struct {
	lo, hi uint32 // range of code points
	script string
}

var chars = make([]Char, MaxChar+1)
var scripts = make(map[string][]Script)
var props = make(map[string][]Script) // a property looks like a script; can share the format

var lastChar uint32 = 0

// In UnicodeData.txt, some ranges are marked like this:
//	3400;<CJK Ideograph Extension A, First>;Lo;0;L;;;;;N;;;;;
//	4DB5;<CJK Ideograph Extension A, Last>;Lo;0;L;;;;;N;;;;;
// parseCategory returns a state variable indicating the weirdness.
type State int

const (
	SNormal State = iota // known to be zero for the type
	SFirst
	SLast
	SMissing
)

func parseCategory(line string) (state State) {
	field := strings.Split(line, ";", -1)
	if len(field) != NumField {
		die.Logf("%5s: %d fields (expected %d)\n", line, len(field), NumField)
	}
	point, err := strconv.Btoui64(field[FCodePoint], 16)
	if err != nil {
		die.Log("%.5s...:", err)
	}
	lastChar = uint32(point)
	if point == 0 {
		return // not interesting and we use 0 as unset
	}
	if point > MaxChar {
		return
	}
	char := &chars[point]
	char.field = field
	if char.codePoint != 0 {
		die.Logf("point U+%04x reused\n")
	}
	char.codePoint = lastChar
	char.category = field[FGeneralCategory]
	category[char.category] = true
	switch char.category {
	case "Nd":
		// Decimal digit
		_, err := strconv.Atoi(field[FNumericValue])
		if err != nil {
			die.Log("U+%04x: bad numeric field: %s", point, err)
		}
	case "Lu":
		char.letter(field[FCodePoint], field[FSimpleLowercaseMapping], field[FSimpleTitlecaseMapping])
	case "Ll":
		char.letter(field[FSimpleUppercaseMapping], field[FCodePoint], field[FSimpleTitlecaseMapping])
	case "Lt":
		char.letter(field[FSimpleUppercaseMapping], field[FSimpleLowercaseMapping], field[FCodePoint])
	case "Lm", "Lo":
		char.letter(field[FSimpleUppercaseMapping], field[FSimpleLowercaseMapping], field[FSimpleTitlecaseMapping])
	}
	switch {
	case strings.Index(field[FName], ", First>") > 0:
		state = SFirst
	case strings.Index(field[FName], ", Last>") > 0:
		state = SLast
	}
	return
}

func (char *Char) dump(s string) {
	fmt.Print(s, " ")
	for i := 0; i < len(char.field); i++ {
		fmt.Printf("%s:%q ", fieldName[i], char.field[i])
	}
	fmt.Print("\n")
}

func (char *Char) letter(u, l, t string) {
	char.upperCase = char.letterValue(u, "U")
	char.lowerCase = char.letterValue(l, "L")
	char.titleCase = char.letterValue(t, "T")
}

func (char *Char) letterValue(s string, cas string) int {
	if s == "" {
		return 0
	}
	v, err := strconv.Btoui64(s, 16)
	if err != nil {
		char.dump(cas)
		die.Logf("U+%04x: bad letter(%s): %s", char.codePoint, s, err)
	}
	return int(v)
}

func allCategories() []string {
	a := make([]string, len(category))
	i := 0
	for k := range category {
		a[i] = k
		i++
	}
	return a
}

func all(scripts map[string][]Script) []string {
	a := make([]string, len(scripts))
	i := 0
	for k := range scripts {
		a[i] = k
		i++
	}
	return a
}

// Extract the version number from the URL
func version() string {
	// Break on slashes and look for the first numeric field
	fields := strings.Split(*url, "/", 0)
	for _, f := range fields {
		if len(f) > 0 && '0' <= f[0] && f[0] <= '9' {
			return f
		}
	}
	die.Log("unknown version")
	return "Unknown"
}

func letterOp(code int) bool {
	switch chars[code].category {
	case "Lu", "Ll", "Lt", "Lm", "Lo":
		return true
	}
	return false
}

func loadChars() {
	if *dataURL == "" {
		flag.Set("data", *url+"UnicodeData.txt")
	}
	resp, _, err := http.Get(*dataURL)
	if err != nil {
		die.Log(err)
	}
	if resp.StatusCode != 200 {
		die.Log("bad GET status for UnicodeData.txt", resp.Status)
	}
	input := bufio.NewReader(resp.Body)
	var first uint32 = 0
	for {
		line, err := input.ReadString('\n')
		if err != nil {
			if err == os.EOF {
				break
			}
			die.Log(err)
		}
		switch parseCategory(line[0 : len(line)-1]) {
		case SNormal:
			if first != 0 {
				die.Logf("bad state normal at U+%04X", lastChar)
			}
		case SFirst:
			if first != 0 {
				die.Logf("bad state first at U+%04X", lastChar)
			}
			first = lastChar
		case SLast:
			if first == 0 {
				die.Logf("bad state last at U+%04X", lastChar)
			}
			for i := first + 1; i <= lastChar; i++ {
				chars[i] = chars[first]
				chars[i].codePoint = i
			}
			first = 0
		}
	}
	resp.Body.Close()
}

func printCategories() {
	if *tablelist == "" {
		return
	}
	// Find out which categories to dump
	list := strings.Split(*tablelist, ",", 0)
	if *tablelist == "all" {
		list = allCategories()
	}
	if *test {
		fullCategoryTest(list)
		return
	}
	fmt.Printf(
		"// Generated by running\n"+
			"//	maketables --tables=%s --data=%s\n"+
			"// DO NOT EDIT\n\n"+
			"package unicode\n\n",
		*tablelist,
		*dataURL)

	fmt.Println("// Version is the Unicode edition from which the tables are derived.")
	fmt.Printf("const Version = %q\n\n", version())

	if *tablelist == "all" {
		fmt.Println("// Categories is the set of Unicode data tables.")
		fmt.Println("var Categories = map[string] []Range {")
		for k, _ := range category {
			fmt.Printf("\t%q: %s,\n", k, k)
		}
		fmt.Printf("}\n\n")
	}

	decl := make(sort.StringArray, len(list))
	ndecl := 0
	for _, name := range list {
		if _, ok := category[name]; !ok {
			die.Log("unknown category", name)
		}
		// We generate an UpperCase name to serve as concise documentation and an _UnderScored
		// name to store the data.  This stops godoc dumping all the tables but keeps them
		// available to clients.
		// Cases deserving special comments
		varDecl := ""
		switch name {
		case "letter":
			varDecl = "\tLetter = letter;	// Letter is the set of Unicode letters.\n"
		case "Nd":
			varDecl = "\tDigit = _Nd;	// Digit is the set of Unicode characters with the \"decimal digit\" property.\n"
		case "Lu":
			varDecl = "\tUpper = _Lu;	// Upper is the set of Unicode upper case letters.\n"
		case "Ll":
			varDecl = "\tLower = _Ll;	// Lower is the set of Unicode lower case letters.\n"
		case "Lt":
			varDecl = "\tTitle = _Lt;	// Title is the set of Unicode title case letters.\n"
		}
		if name != "letter" {
			varDecl += fmt.Sprintf(
				"\t%s = _%s;	// %s is the set of Unicode characters in category %s.\n",
				name, name, name, name)
		}
		decl[ndecl] = varDecl
		ndecl++
		if name == "letter" { // special case
			dumpRange(
				"var letter = []Range {\n",
				letterOp)
			continue
		}
		dumpRange(
			fmt.Sprintf("var _%s = []Range {\n", name),
			func(code int) bool { return chars[code].category == name })
	}
	decl.Sort()
	fmt.Println("var (")
	for _, d := range decl {
		fmt.Print(d)
	}
	fmt.Println(")\n")
}

type Op func(code int) bool

const format = "\tRange{0x%04x, 0x%04x, %d},\n"

func dumpRange(header string, inCategory Op) {
	fmt.Print(header)
	next := 0
	// one Range for each iteration
	for {
		// look for start of range
		for next < len(chars) && !inCategory(next) {
			next++
		}
		if next >= len(chars) {
			// no characters remain
			break
		}

		// start of range
		lo := next
		hi := next
		stride := 1
		// accept lo
		next++
		// look for another character to set the stride
		for next < len(chars) && !inCategory(next) {
			next++
		}
		if next >= len(chars) {
			// no more characters
			fmt.Printf(format, lo, hi, stride)
			break
		}
		// set stride
		stride = next - lo
		// check for length of run. next points to first jump in stride
		for i := next; i < len(chars); i++ {
			if inCategory(i) == (((i - lo) % stride) == 0) {
				// accept
				if inCategory(i) {
					hi = i
				}
			} else {
				// no more characters in this run
				break
			}
		}
		fmt.Printf(format, lo, hi, stride)
		// next range: start looking where this range ends
		next = hi + 1
	}
	fmt.Print("}\n\n")
}

func fullCategoryTest(list []string) {
	for _, name := range list {
		if _, ok := category[name]; !ok {
			die.Log("unknown category", name)
		}
		r, ok := unicode.Categories[name]
		if !ok {
			die.Log("unknown table", name)
		}
		if name == "letter" {
			verifyRange(name, letterOp, r)
		} else {
			verifyRange(
				name,
				func(code int) bool { return chars[code].category == name },
				r)
		}
	}
}

func verifyRange(name string, inCategory Op, table []unicode.Range) {
	for i := range chars {
		web := inCategory(i)
		pkg := unicode.Is(table, i)
		if web != pkg {
			fmt.Fprintf(os.Stderr, "%s: U+%04X: web=%t pkg=%t\n", name, i, web, pkg)
		}
	}
}

func parseScript(line string, scripts map[string][]Script) {
	comment := strings.Index(line, "#")
	if comment >= 0 {
		line = line[0:comment]
	}
	line = strings.TrimSpace(line)
	if len(line) == 0 {
		return
	}
	field := strings.Split(line, ";", -1)
	if len(field) != 2 {
		die.Logf("%s: %d fields (expected 2)\n", line, len(field))
	}
	matches := scriptRe.MatchStrings(line)
	if len(matches) != 4 {
		die.Logf("%s: %d matches (expected 3)\n", line, len(matches))
	}
	lo, err := strconv.Btoui64(matches[1], 16)
	if err != nil {
		die.Log("%.5s...:", err)
	}
	hi := lo
	if len(matches[2]) > 2 { // ignore leading ..
		hi, err = strconv.Btoui64(matches[2][2:], 16)
		if err != nil {
			die.Log("%.5s...:", err)
		}
	}
	name := matches[3]
	s, ok := scripts[name]
	if !ok || len(s) == cap(s) {
		ns := make([]Script, len(s), len(s)+100)
		for i, sc := range s {
			ns[i] = sc
		}
		s = ns
	}
	s = s[0 : len(s)+1]
	s[len(s)-1] = Script{uint32(lo), uint32(hi), name}
	scripts[name] = s
}

// The script tables have a lot of adjacent elements. Fold them together.
func foldAdjacent(r []Script) []unicode.Range {
	s := make([]unicode.Range, 0, len(r))
	j := 0
	for i := 0; i < len(r); i++ {
		if j > 0 && int(r[i].lo) == s[j-1].Hi+1 {
			s[j-1].Hi = int(r[i].hi)
		} else {
			s = s[0 : j+1]
			s[j] = unicode.Range{int(r[i].lo), int(r[i].hi), 1}
			j++
		}
	}
	return s
}

func fullScriptTest(list []string, installed map[string][]unicode.Range, scripts map[string][]Script) {
	for _, name := range list {
		if _, ok := scripts[name]; !ok {
			die.Log("unknown script", name)
		}
		_, ok := installed[name]
		if !ok {
			die.Log("unknown table", name)
		}
		for _, script := range scripts[name] {
			for r := script.lo; r <= script.hi; r++ {
				if !unicode.Is(installed[name], int(r)) {
					fmt.Fprintf(os.Stderr, "U+%04X: not in script %s\n", r, name)
				}
			}
		}
	}
}

// PropList.txt has the same format as Scripts.txt so we can share its parser.
func printScriptOrProperty(doProps bool) {
	flag := "scripts"
	flaglist := *scriptlist
	file := "Scripts.txt"
	table := scripts
	installed := unicode.Scripts
	if doProps {
		flag = "props"
		flaglist = *proplist
		file = "PropList.txt"
		table = props
		installed = unicode.Properties
	}
	if flaglist == "" {
		return
	}
	var err os.Error
	resp, _, err := http.Get(*url + file)
	if err != nil {
		die.Log(err)
	}
	if resp.StatusCode != 200 {
		die.Log("bad GET status for ", file, ":", resp.Status)
	}
	input := bufio.NewReader(resp.Body)
	for {
		line, err := input.ReadString('\n')
		if err != nil {
			if err == os.EOF {
				break
			}
			die.Log(err)
		}
		parseScript(line[0:len(line)-1], table)
	}
	resp.Body.Close()

	// Find out which scripts to dump
	list := strings.Split(flaglist, ",", 0)
	if flaglist == "all" {
		list = all(table)
	}
	if *test {
		fullScriptTest(list, installed, table)
		return
	}

	fmt.Printf(
		"// Generated by running\n"+
			"//	maketables --%s=%s --url=%s\n"+
			"// DO NOT EDIT\n\n",
		flag,
		flaglist,
		*url)
	if flaglist == "all" {
		if doProps {
			fmt.Println("// Properties is the set of Unicode property tables.")
			fmt.Println("var Properties = map[string] []Range {")
		} else {
			fmt.Println("// Scripts is the set of Unicode script tables.")
			fmt.Println("var Scripts = map[string] []Range {")
		}
		for k, _ := range table {
			fmt.Printf("\t%q: %s,\n", k, k)
		}
		fmt.Printf("}\n\n")
	}

	decl := make(sort.StringArray, len(list))
	ndecl := 0
	for _, name := range list {
		if doProps {
			decl[ndecl] = fmt.Sprintf(
				"\t%s = _%s;\t// %s is the set of Unicode characters with property %s.\n",
				name, name, name, name)
		} else {
			decl[ndecl] = fmt.Sprintf(
				"\t%s = _%s;\t// %s is the set of Unicode characters in script %s.\n",
				name, name, name, name)
		}
		ndecl++
		fmt.Printf("var _%s = []Range {\n", name)
		ranges := foldAdjacent(table[name])
		for _, s := range ranges {
			fmt.Printf(format, s.Lo, s.Hi, s.Stride)
		}
		fmt.Printf("}\n\n")
	}
	decl.Sort()
	fmt.Println("var (")
	for _, d := range decl {
		fmt.Print(d)
	}
	fmt.Println(")\n")
}

const (
	CaseUpper = 1 << iota
	CaseLower
	CaseTitle
	CaseNone    = 0  // must be zero
	CaseMissing = -1 // character not present; not a valid case state
)

type caseState struct {
	point        int
	_case        int
	deltaToUpper int
	deltaToLower int
	deltaToTitle int
}

// Is d a continuation of the state of c?
func (c *caseState) adjacent(d *caseState) bool {
	if d.point < c.point {
		c, d = d, c
	}
	switch {
	case d.point != c.point+1: // code points not adjacent (shouldn't happen)
		return false
	case d._case != c._case: // different cases
		return c.upperLowerAdjacent(d)
	case c._case == CaseNone:
		return false
	case c._case == CaseMissing:
		return false
	case d.deltaToUpper != c.deltaToUpper:
		return false
	case d.deltaToLower != c.deltaToLower:
		return false
	case d.deltaToTitle != c.deltaToTitle:
		return false
	}
	return true
}

// Is d the same as c, but opposite in upper/lower case? this would make it
// an element of an UpperLower sequence.
func (c *caseState) upperLowerAdjacent(d *caseState) bool {
	// check they're a matched case pair.  we know they have adjacent values
	switch {
	case c._case == CaseUpper && d._case != CaseLower:
		return false
	case c._case == CaseLower && d._case != CaseUpper:
		return false
	}
	// matched pair (at least in upper/lower).  make the order Upper Lower
	if c._case == CaseLower {
		c, d = d, c
	}
	// for an Upper Lower sequence the deltas have to be in order
	//	c: 0 1 0
	//	d: -1 0 -1
	switch {
	case c.deltaToUpper != 0:
		return false
	case c.deltaToLower != 1:
		return false
	case c.deltaToTitle != 0:
		return false
	case d.deltaToUpper != -1:
		return false
	case d.deltaToLower != 0:
		return false
	case d.deltaToTitle != -1:
		return false
	}
	return true
}

// Does this character start an UpperLower sequence?
func (c *caseState) isUpperLower() bool {
	// for an Upper Lower sequence the deltas have to be in order
	//	c: 0 1 0
	switch {
	case c.deltaToUpper != 0:
		return false
	case c.deltaToLower != 1:
		return false
	case c.deltaToTitle != 0:
		return false
	}
	return true
}

// Does this character start a LowerUpper sequence?
func (c *caseState) isLowerUpper() bool {
	// for an Upper Lower sequence the deltas have to be in order
	//	c: -1 0 -1
	switch {
	case c.deltaToUpper != -1:
		return false
	case c.deltaToLower != 0:
		return false
	case c.deltaToTitle != -1:
		return false
	}
	return true
}

func getCaseState(i int) (c *caseState) {
	c = &caseState{point: i, _case: CaseNone}
	ch := &chars[i]
	switch int(ch.codePoint) {
	case 0:
		c._case = CaseMissing // Will get NUL wrong but that doesn't matter
		return
	case ch.upperCase:
		c._case = CaseUpper
	case ch.lowerCase:
		c._case = CaseLower
	case ch.titleCase:
		c._case = CaseTitle
	}
	if ch.upperCase != 0 {
		c.deltaToUpper = ch.upperCase - i
	}
	if ch.lowerCase != 0 {
		c.deltaToLower = ch.lowerCase - i
	}
	if ch.titleCase != 0 {
		c.deltaToTitle = ch.titleCase - i
	}
	return
}

func printCases() {
	if !*cases {
		return
	}
	if *test {
		fullCaseTest()
		return
	}
	fmt.Printf(
		"// Generated by running\n"+
			"//	maketables --data=%s\n"+
			"// DO NOT EDIT\n\n"+
			"// CaseRanges is the table describing case mappings for all letters with\n"+
			"// non-self mappings.\n"+
			"var CaseRanges = _CaseRanges\n"+
			"var _CaseRanges = []CaseRange {\n",
		*dataURL)

	var startState *caseState    // the start of a run; nil for not active
	var prevState = &caseState{} // the state of the previous character
	for i := range chars {
		state := getCaseState(i)
		if state.adjacent(prevState) {
			prevState = state
			continue
		}
		// end of run (possibly)
		printCaseRange(startState, prevState)
		startState = nil
		if state._case != CaseMissing && state._case != CaseNone {
			startState = state
		}
		prevState = state
	}
	fmt.Printf("}\n")
}

func printCaseRange(lo, hi *caseState) {
	if lo == nil {
		return
	}
	if lo.deltaToUpper == 0 && lo.deltaToLower == 0 && lo.deltaToTitle == 0 {
		// character represents itself in all cases - no need to mention it
		return
	}
	switch {
	case hi.point > lo.point && lo.isUpperLower():
		fmt.Printf("\tCaseRange{0x%04X, 0x%04X, d{UpperLower, UpperLower, UpperLower}},\n",
			lo.point, hi.point)
	case hi.point > lo.point && lo.isLowerUpper():
		die.Log("LowerUpper sequence: should not happen: U+%04X.  If it's real, need to fix To()", lo.point)
		fmt.Printf("\tCaseRange{0x%04X, 0x%04X, d{LowerUpper, LowerUpper, LowerUpper}},\n",
			lo.point, hi.point)
	default:
		fmt.Printf("\tCaseRange{0x%04X, 0x%04X, d{%d, %d, %d}},\n",
			lo.point, hi.point,
			lo.deltaToUpper, lo.deltaToLower, lo.deltaToTitle)
	}
}

// If the cased value in the Char is 0, it means use the rune itself.
func caseIt(rune, cased int) int {
	if cased == 0 {
		return rune
	}
	return cased
}

func fullCaseTest() {
	for i, c := range chars {
		lower := unicode.ToLower(i)
		want := caseIt(i, c.lowerCase)
		if lower != want {
			fmt.Fprintf(os.Stderr, "lower U+%04X should be U+%04X is U+%04X\n", i, want, lower)
		}
		upper := unicode.ToUpper(i)
		want = caseIt(i, c.upperCase)
		if upper != want {
			fmt.Fprintf(os.Stderr, "upper U+%04X should be U+%04X is U+%04X\n", i, want, upper)
		}
		title := unicode.ToTitle(i)
		want = caseIt(i, c.titleCase)
		if title != want {
			fmt.Fprintf(os.Stderr, "title U+%04X should be U+%04X is U+%04X\n", i, want, title)
		}
	}
}
