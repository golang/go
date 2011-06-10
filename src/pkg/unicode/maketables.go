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
	printLatinProperties()
	printSizes()
}

var dataURL = flag.String("data", "", "full URL for UnicodeData.txt; defaults to --url/UnicodeData.txt")
var url = flag.String("url",
	"http://www.unicode.org/Public/6.0.0/ucd/",
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

var scriptRe = regexp.MustCompile(`^([0-9A-F]+)(\.\.[0-9A-F]+)? *; ([A-Za-z_]+)$`)
var logger = log.New(os.Stderr, "", log.Lshortfile)

var category = map[string]bool{
	// Nd Lu etc.
	// We use one-character names to identify merged categories
	"L": true, // Lu Ll Lt Lm Lo
	"P": true, // Pc Pd Ps Pe Pu Pf Po
	"M": true, // Mn Mc Me
	"N": true, // Nd Nl No
	"S": true, // Sm Sc Sk So
	"Z": true, // Zs Zl Zp
	"C": true, // Cc Cf Cs Co Cn
}

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
		logger.Fatalf("%5s: %d fields (expected %d)\n", line, len(field), NumField)
	}
	point, err := strconv.Btoui64(field[FCodePoint], 16)
	if err != nil {
		logger.Fatalf("%.5s...: %s", line, err)
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
		logger.Fatalf("point %U reused", point)
	}
	char.codePoint = lastChar
	char.category = field[FGeneralCategory]
	category[char.category] = true
	switch char.category {
	case "Nd":
		// Decimal digit
		_, err := strconv.Atoi(field[FNumericValue])
		if err != nil {
			logger.Fatalf("%U: bad numeric field: %s", point, err)
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
		logger.Fatalf("%U: bad letter(%s): %s", char.codePoint, s, err)
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
	fields := strings.Split(*url, "/", -1)
	for _, f := range fields {
		if len(f) > 0 && '0' <= f[0] && f[0] <= '9' {
			return f
		}
	}
	logger.Fatal("unknown version")
	return "Unknown"
}

func categoryOp(code int, class uint8) bool {
	category := chars[code].category
	return len(category) > 0 && category[0] == class
}

func loadChars() {
	if *dataURL == "" {
		flag.Set("data", *url+"UnicodeData.txt")
	}
	resp, err := http.Get(*dataURL)
	if err != nil {
		logger.Fatal(err)
	}
	if resp.StatusCode != 200 {
		logger.Fatal("bad GET status for UnicodeData.txt", resp.Status)
	}
	input := bufio.NewReader(resp.Body)
	var first uint32 = 0
	for {
		line, err := input.ReadString('\n')
		if err != nil {
			if err == os.EOF {
				break
			}
			logger.Fatal(err)
		}
		switch parseCategory(line[0 : len(line)-1]) {
		case SNormal:
			if first != 0 {
				logger.Fatalf("bad state normal at %U", lastChar)
			}
		case SFirst:
			if first != 0 {
				logger.Fatalf("bad state first at %U", lastChar)
			}
			first = lastChar
		case SLast:
			if first == 0 {
				logger.Fatalf("bad state last at %U", lastChar)
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

const progHeader = `// Generated by running
//	maketables --tables=%s --data=%s
// DO NOT EDIT

package unicode

`


func printCategories() {
	if *tablelist == "" {
		return
	}
	// Find out which categories to dump
	list := strings.Split(*tablelist, ",", -1)
	if *tablelist == "all" {
		list = allCategories()
	}
	if *test {
		fullCategoryTest(list)
		return
	}
	fmt.Printf(progHeader, *tablelist, *dataURL)

	fmt.Println("// Version is the Unicode edition from which the tables are derived.")
	fmt.Printf("const Version = %q\n\n", version())

	if *tablelist == "all" {
		fmt.Println("// Categories is the set of Unicode data tables.")
		fmt.Println("var Categories = map[string] *RangeTable {")
		for k := range category {
			fmt.Printf("\t%q: %s,\n", k, k)
		}
		fmt.Print("}\n\n")
	}

	decl := make(sort.StringSlice, len(list))
	ndecl := 0
	for _, name := range list {
		if _, ok := category[name]; !ok {
			logger.Fatal("unknown category", name)
		}
		// We generate an UpperCase name to serve as concise documentation and an _UnderScored
		// name to store the data.  This stops godoc dumping all the tables but keeps them
		// available to clients.
		// Cases deserving special comments
		varDecl := ""
		switch name {
		case "C":
			varDecl = "\tOther = _C;	// Other/C is the set of Unicode control and special characters, category C.\n"
			varDecl += "\tC = _C\n"
		case "L":
			varDecl = "\tLetter = _L;	// Letter/L is the set of Unicode letters, category L.\n"
			varDecl += "\tL = _L\n"
		case "M":
			varDecl = "\tMark = _M;	// Mark/M is the set of Unicode mark characters, category  M.\n"
			varDecl += "\tM = _M\n"
		case "N":
			varDecl = "\tNumber = _N;	// Number/N is the set of Unicode number characters, category N.\n"
			varDecl += "\tN = _N\n"
		case "P":
			varDecl = "\tPunct = _P;	// Punct/P is the set of Unicode punctuation characters, category P.\n"
			varDecl += "\tP = _P\n"
		case "S":
			varDecl = "\tSymbol = _S;	// Symbol/S is the set of Unicode symbol characters, category S.\n"
			varDecl += "\tS = _S\n"
		case "Z":
			varDecl = "\tSpace = _Z;	// Space/Z is the set of Unicode space characters, category Z.\n"
			varDecl += "\tZ = _Z\n"
		case "Nd":
			varDecl = "\tDigit = _Nd;	// Digit is the set of Unicode characters with the \"decimal digit\" property.\n"
		case "Lu":
			varDecl = "\tUpper = _Lu;	// Upper is the set of Unicode upper case letters.\n"
		case "Ll":
			varDecl = "\tLower = _Ll;	// Lower is the set of Unicode lower case letters.\n"
		case "Lt":
			varDecl = "\tTitle = _Lt;	// Title is the set of Unicode title case letters.\n"
		}
		if len(name) > 1 {
			varDecl += fmt.Sprintf(
				"\t%s = _%s;	// %s is the set of Unicode characters in category %s.\n",
				name, name, name, name)
		}
		decl[ndecl] = varDecl
		ndecl++
		if len(name) == 1 { // unified categories
			decl := fmt.Sprintf("var _%s = &RangeTable{\n", name)
			dumpRange(
				decl,
				func(code int) bool { return categoryOp(code, name[0]) })
			continue
		}
		dumpRange(
			fmt.Sprintf("var _%s = &RangeTable{\n", name),
			func(code int) bool { return chars[code].category == name })
	}
	decl.Sort()
	fmt.Println("var (")
	for _, d := range decl {
		fmt.Print(d)
	}
	fmt.Print(")\n\n")
}

type Op func(code int) bool

const format = "\t\t{0x%04x, 0x%04x, %d},\n"

func dumpRange(header string, inCategory Op) {
	fmt.Print(header)
	next := 0
	fmt.Print("\tR16: []Range16{\n")
	// one Range for each iteration
	count := &range16Count
	size := 16
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
		size, count = printRange(uint32(lo), uint32(hi), uint32(stride), size, count)
		// next range: start looking where this range ends
		next = hi + 1
	}
	fmt.Print("\t},\n")
	fmt.Print("}\n\n")
}

func printRange(lo, hi, stride uint32, size int, count *int) (int, *int) {
	if size == 16 && hi >= 1<<16 {
		if lo < 1<<16 {
			if lo+stride != hi {
				logger.Fatalf("unexpected straddle: %U %U %d", lo, hi, stride)
			}
			// No range contains U+FFFF as an instance, so split
			// the range into two entries. That way we can maintain
			// the invariant that R32 contains only >= 1<<16.
			fmt.Printf(format, lo, lo, 1)
			lo = hi
			stride = 1
			*count++
		}
		fmt.Print("\t},\n")
		fmt.Print("\tR32: []Range32{\n")
		size = 32
		count = &range32Count
	}
	fmt.Printf(format, lo, hi, stride)
	*count++
	return size, count
}

func fullCategoryTest(list []string) {
	for _, name := range list {
		if _, ok := category[name]; !ok {
			logger.Fatal("unknown category", name)
		}
		r, ok := unicode.Categories[name]
		if !ok && len(name) > 1 {
			logger.Fatalf("unknown table %q", name)
		}
		if len(name) == 1 {
			verifyRange(name, func(code int) bool { return categoryOp(code, name[0]) }, r)
		} else {
			verifyRange(
				name,
				func(code int) bool { return chars[code].category == name },
				r)
		}
	}
}

func verifyRange(name string, inCategory Op, table *unicode.RangeTable) {
	count := 0
	for i := range chars {
		web := inCategory(i)
		pkg := unicode.Is(table, i)
		if web != pkg {
			fmt.Fprintf(os.Stderr, "%s: %U: web=%t pkg=%t\n", name, i, web, pkg)
			count++
			if count > 10 {
				break
			}
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
		logger.Fatalf("%s: %d fields (expected 2)\n", line, len(field))
	}
	matches := scriptRe.FindStringSubmatch(line)
	if len(matches) != 4 {
		logger.Fatalf("%s: %d matches (expected 3)\n", line, len(matches))
	}
	lo, err := strconv.Btoui64(matches[1], 16)
	if err != nil {
		logger.Fatalf("%.5s...: %s", line, err)
	}
	hi := lo
	if len(matches[2]) > 2 { // ignore leading ..
		hi, err = strconv.Btoui64(matches[2][2:], 16)
		if err != nil {
			logger.Fatalf("%.5s...: %s", line, err)
		}
	}
	name := matches[3]
	scripts[name] = append(scripts[name], Script{uint32(lo), uint32(hi), name})
}

// The script tables have a lot of adjacent elements. Fold them together.
func foldAdjacent(r []Script) []unicode.Range32 {
	s := make([]unicode.Range32, 0, len(r))
	j := 0
	for i := 0; i < len(r); i++ {
		if j > 0 && r[i].lo == s[j-1].Hi+1 {
			s[j-1].Hi = r[i].hi
		} else {
			s = s[0 : j+1]
			s[j] = unicode.Range32{uint32(r[i].lo), uint32(r[i].hi), 1}
			j++
		}
	}
	return s
}

func fullScriptTest(list []string, installed map[string]*unicode.RangeTable, scripts map[string][]Script) {
	for _, name := range list {
		if _, ok := scripts[name]; !ok {
			logger.Fatal("unknown script", name)
		}
		_, ok := installed[name]
		if !ok {
			logger.Fatal("unknown table", name)
		}
		for _, script := range scripts[name] {
			for r := script.lo; r <= script.hi; r++ {
				if !unicode.Is(installed[name], int(r)) {
					fmt.Fprintf(os.Stderr, "%U: not in script %s\n", r, name)
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
	resp, err := http.Get(*url + file)
	if err != nil {
		logger.Fatal(err)
	}
	if resp.StatusCode != 200 {
		logger.Fatal("bad GET status for ", file, ":", resp.Status)
	}
	input := bufio.NewReader(resp.Body)
	for {
		line, err := input.ReadString('\n')
		if err != nil {
			if err == os.EOF {
				break
			}
			logger.Fatal(err)
		}
		parseScript(line[0:len(line)-1], table)
	}
	resp.Body.Close()

	// Find out which scripts to dump
	list := strings.Split(flaglist, ",", -1)
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
			fmt.Println("var Properties = map[string] *RangeTable{")
		} else {
			fmt.Println("// Scripts is the set of Unicode script tables.")
			fmt.Println("var Scripts = map[string] *RangeTable{")
		}
		for k := range table {
			fmt.Printf("\t%q: %s,\n", k, k)
		}
		fmt.Print("}\n\n")
	}

	decl := make(sort.StringSlice, len(list))
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
		fmt.Printf("var _%s = &RangeTable {\n", name)
		fmt.Print("\tR16: []Range16{\n")
		ranges := foldAdjacent(table[name])
		size := 16
		count := &range16Count
		for _, s := range ranges {
			size, count = printRange(s.Lo, s.Hi, s.Stride, size, count)
		}
		fmt.Print("\t},\n")
		fmt.Print("}\n\n")
	}
	decl.Sort()
	fmt.Println("var (")
	for _, d := range decl {
		fmt.Print(d)
	}
	fmt.Print(")\n\n")
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
	fmt.Print("}\n")
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
		fmt.Printf("\t{0x%04X, 0x%04X, d{UpperLower, UpperLower, UpperLower}},\n",
			lo.point, hi.point)
	case hi.point > lo.point && lo.isLowerUpper():
		logger.Fatalf("LowerUpper sequence: should not happen: %U.  If it's real, need to fix To()", lo.point)
		fmt.Printf("\t{0x%04X, 0x%04X, d{LowerUpper, LowerUpper, LowerUpper}},\n",
			lo.point, hi.point)
	default:
		fmt.Printf("\t{0x%04X, 0x%04X, d{%d, %d, %d}},\n",
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
			fmt.Fprintf(os.Stderr, "lower %U should be %U is %U\n", i, want, lower)
		}
		upper := unicode.ToUpper(i)
		want = caseIt(i, c.upperCase)
		if upper != want {
			fmt.Fprintf(os.Stderr, "upper %U should be %U is %U\n", i, want, upper)
		}
		title := unicode.ToTitle(i)
		want = caseIt(i, c.titleCase)
		if title != want {
			fmt.Fprintf(os.Stderr, "title %U should be %U is %U\n", i, want, title)
		}
	}
}

func printLatinProperties() {
	if *test {
		return
	}
	fmt.Println("var properties = [MaxLatin1+1]uint8{")
	for code := 0; code <= unicode.MaxLatin1; code++ {
		var property string
		switch chars[code].category {
		case "Cc", "": // NUL has no category.
			property = "pC"
		case "Cf": // soft hyphen, unique category, not printable.
			property = "0"
		case "Ll":
			property = "pLl | pp"
		case "Lu":
			property = "pLu | pp"
		case "Nd", "No":
			property = "pN | pp"
		case "Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps":
			property = "pP | pp"
		case "Sc", "Sk", "Sm", "So":
			property = "pS | pp"
		case "Zs":
			property = "pZ"
		default:
			logger.Fatalf("%U has unknown category %q", code, chars[code].category)
		}
		// Special case
		if code == ' ' {
			property = "pZ | pp"
		}
		fmt.Printf("\t0x%.2X: %s, // %q\n", code, property, code)
	}
	fmt.Println("}")
}

var range16Count = 0 // Number of entries in the 16-bit range tables.
var range32Count = 0 // Number of entries in the 32-bit range tables.

func printSizes() {
	if *test {
		return
	}
	fmt.Println()
	fmt.Printf("// Range entries: %d 16-bit, %d 32-bit, %d total.\n", range16Count, range32Count, range16Count+range32Count)
	range16Bytes := range16Count * 3 * 2
	range32Bytes := range32Count * 3 * 4
	fmt.Printf("// Range bytes: %d 16-bit, %d 32-bit, %d total.\n", range16Bytes, range32Bytes, range16Bytes+range32Bytes)
}
