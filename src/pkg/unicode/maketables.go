// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Unicode table generator.
// Data read from the web.

package main

import (
	"bufio";
	"flag";
	"fmt";
	"http";
	"log";
	"os";
	"sort";
	"strconv";
	"strings";
	"regexp";
	"unicode";
)

var dataUrl = flag.String("data", "", "full URL for UnicodeData.txt; defaults to --url/UnicodeData.txt");
var url = flag.String("url",
	"http://www.unicode.org/Public/5.1.0/ucd/",
	"URL of Unicode database directory")
var tablelist = flag.String("tables",
	"all",
	"comma-separated list of which tables to generate; can be letter");
var scriptlist = flag.String("scripts",
	"all",
	"comma-separated list of which script tables to generate");
var test = flag.Bool("test",
	false,
	"test existing tables; can be used to compare web data with package data");
var scriptRe *regexp.Regexp

var die = log.New(os.Stderr, nil, "", log.Lexit|log.Lshortfile);

var category = map[string] bool{ "letter":true }	// Nd Lu etc. letter is a special case

// UnicodeData.txt has form:
//	0037;DIGIT SEVEN;Nd;0;EN;;7;7;7;N;;;;;
//	007A;LATIN SMALL LETTER Z;Ll;0;L;;;;;N;;;005A;;005A
// See http://www.unicode.org/Public/5.1.0/ucd/UCD.html for full explanation
// The fields
const (
	FCodePoint = iota;
	FName;
	FGeneralCategory;
	FCanonicalCombiningClass;
	FBidiClass;
	FDecompositionType;
	FDecompositionMapping;
	FNumericType;
	FNumericValue;
	FBidiMirrored;
	FUnicode1Name;
	FISOComment;
	FSimpleUppercaseMapping;
	FSimpleLowercaseMapping;
	FSimpleTitlecaseMapping;
	NumField;

	MaxChar = 0x10FFFF;	// anything above this shouldn't exist
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
	"SimpleTitlecaseMapping"
}

// This contains only the properties we're interested in.
type Char struct {
	field	[]string; 	// debugging only; could be deleted if we take out char.dump()
	codePoint	uint32;	// redundant (it's the index in the chars table) but useful
	category	string;
	upperCase	uint32;
	lowerCase	uint32;
	titleCase	uint32;
}

// Scripts.txt has form:
//	A673          ; Cyrillic # Po       SLAVONIC ASTERISK
//	A67C..A67D    ; Cyrillic # Mn   [2] COMBINING CYRILLIC KAVYKA..COMBINING CYRILLIC PAYEROK
// See http://www.unicode.org/Public/5.1.0/ucd/UCD.html for full explanation

type Script struct {
	lo, hi	uint32;	// range of code points
	script	string;
}

func main() {
	flag.Parse();
	printCategories();
	printScripts();
}

var chars = make([]Char, MaxChar)
var scripts = make(map[string] []Script)

var lastChar uint32 = 0;

// In UnicodeData.txt, some ranges are marked like this:
// 3400;<CJK Ideograph Extension A, First>;Lo;0;L;;;;;N;;;;;
// 4DB5;<CJK Ideograph Extension A, Last>;Lo;0;L;;;;;N;;;;;
// parseCategory returns a state variable indicating the weirdness.
type State int
const (
	SNormal State = iota;	// known to be zero for the type
	SFirst;
	SLast;
)

func parseCategory(line string) (state State) {
	field := strings.Split(line, ";", -1);
	if len(field) != NumField {
		die.Logf("%5s: %d fields (expected %d)\n", line, len(field), NumField);
	}
	point, err := strconv.Btoui64(field[FCodePoint], 16);
	if err != nil {
		die.Log("%.5s...:", err)
	}
	lastChar = uint32(point);
	if point == 0 {
		return	// not interesting and we use 0 as unset
	}
	if point >= MaxChar {
		return;
	}
	char := &chars[point];
	char.field=field;
	if char.codePoint != 0 {
		die.Logf("point U+%04x reused\n");
	}
	char.codePoint = lastChar;
	char.category = field[FGeneralCategory];
	category[char.category] = true;
	switch char.category {
	case "Nd":
		// Decimal digit
		v, err := strconv.Atoi(field[FNumericValue]);
		if err != nil {
			die.Log("U+%04x: bad numeric field: %s", point, err);
		}
	case "Lu":
		char.letter(field[FCodePoint], field[FSimpleLowercaseMapping], field[FSimpleTitlecaseMapping]);
	case "Ll":
		char.letter(field[FSimpleUppercaseMapping], field[FCodePoint], field[FSimpleTitlecaseMapping]);
	case "Lt":
		char.letter(field[FSimpleUppercaseMapping], field[FSimpleLowercaseMapping], field[FCodePoint]);
	case "Lm", "Lo":
		char.letter(field[FSimpleUppercaseMapping], field[FSimpleLowercaseMapping], field[FSimpleTitlecaseMapping]);
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
	fmt.Print(s, " ");
	for i:=0;i<len(char.field);i++ {
		fmt.Printf("%s:%q ", fieldName[i], char.field[i]);
	}
	fmt.Print("\n");
}

func (char *Char) letter(u, l, t string) {
	char.upperCase = char.letterValue(u, "U");
	char.lowerCase = char.letterValue(l, "L");
	char.titleCase = char.letterValue(t, "T");
}

func (char *Char) letterValue(s string, cas string) uint32 {
	if s == "" {
		return 0
	}
	v, err := strconv.Btoui64(s, 16);
	if err != nil {
		char.dump(cas);
		die.Logf("U+%04x: bad letter(%s): %s", char.codePoint, s, err)
	}
	return uint32(v)
}

func allCategories() []string {
	a := make([]string, len(category));
	i := 0;
	for k := range category {
		a[i] = k;
		i++;
	}
	return a;
}

func allScripts() []string {
	a := make([]string, len(scripts));
	i := 0;
	for k := range scripts {
		a[i] = k;
		i++;
	}
	return a;
}

// Extract the version number from the URL
func version() string {
	// Break on slashes and look for the first numeric field
	fields := strings.Split(*url, "/", 0);
	for _, f := range fields {
		if len(f) > 0 && '0' <= f[0] && f[0] <= '9' {
			return f
		}
	}
	die.Log("unknown version");
	return "Unknown";
}

func letterOp(code int) bool {
	switch chars[code].category {
	case "Lu", "Ll", "Lt", "Lm", "Lo":
		return true
	}
	return false
}

func printCategories() {
	if *tablelist == "" {
		return
	}
	if *dataUrl == "" {
		flag.Set("data", *url + "UnicodeData.txt");
	}
	resp, _, err := http.Get(*dataUrl);
	if err != nil {
		die.Log(err);
	}
	if resp.StatusCode != 200 {
		die.Log("bad GET status for UnicodeData.txt", resp.Status);
	}
	input := bufio.NewReader(resp.Body);
	var first uint32 = 0;
	for {
		line, err := input.ReadString('\n');
		if err != nil {
			if err == os.EOF {
				break;
			}
			die.Log(err);
		}
		switch parseCategory(line[0:len(line)-1]) {
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
			for i := first+1; i <= lastChar; i++ {
				chars[i] = chars[first];
				chars[i].codePoint = i;
			}
			first = 0
		}
	}
	resp.Body.Close();
	// Find out which categories to dump
	list := strings.Split(*tablelist, ",", 0);
	if *tablelist == "all" {
		list = allCategories()
	}
	if *test {
		fullCategoryTest(list);
		return
	}
	fmt.Printf(
		"// Generated by running\n"
		"//	maketables --tables=%s --url=%s\n"
		"// DO NOT EDIT\n\n"
		"package unicode\n\n",
		*tablelist,
		*url
	);

	fmt.Println("// Version is the Unicode edition from which the tables are derived.");
	fmt.Printf("const Version = %q\n\n", version());

	if *tablelist == "all" {
		fmt.Println("// Categories is the set of Unicode data tables.");
			fmt.Println("var Categories = map[string] []Range {");
		for k, _ := range category {
			fmt.Printf("\t%q: %s,\n", k, k);
		}
		fmt.Printf("}\n\n");
	}

	decl := make(sort.StringArray, len(list));
	ndecl := 0;
	for _, name := range list {
		if _, ok := category[name]; !ok {
			die.Log("unknown category", name);
		}
		// We generate an UpperCase name to serve as concise documentation and an _UnderScored
		// name to store the data.  This stops godoc dumping all the tables but keeps them
		// available to clients.
		// Cases deserving special comments
		varDecl := "";
		switch name {
		case "letter":
			varDecl = "\tLetter = letter;	// Letter is the set of Unicode letters.\n";
		case "Nd":
			varDecl = "\tDigit = _Nd;	// Digit is the set of Unicode characters with the \"decimal digit\" property.\n";
		case "Lu":
			varDecl = "\tUpper = _Lu;	// Upper is the set of Unicode upper case letters.\n";
		case "Ll":
			varDecl = "\tLower = _Ll;	// Lower is the set of Unicode lower case letters.\n";
		case "Lt":
			varDecl = "\tTitle = _Lt;	// Title is the set of Unicode title case letters.\n";
		}
		if name != "letter" {
			varDecl += fmt.Sprintf(
				"\t%s = _%s;	// %s is the set of Unicode characters in category %s.\n",
				name, name, name, name
			);
		}
		decl[ndecl] = varDecl;
		ndecl++;
		if name == "letter" {	// special case
			dumpRange(
				"var letter = []Range {\n",
				letterOp
			);
			continue;
		}
		dumpRange(
			fmt.Sprintf("var _%s = []Range {\n", name),
			func(code int) bool { return chars[code].category == name }
		);
	}
	decl.Sort();
	fmt.Println("var (");
	for _, d := range decl {
		fmt.Print(d);
	}
	fmt.Println(")\n");
}

type Op func(code int) bool
const format = "\tRange{0x%04x, 0x%04x, %d},\n";

func dumpRange(header string, inCategory Op) {
	fmt.Print(header);
	next := 0;
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
		lo := next;
		hi := next;
		stride := 1;
		// accept lo
		next++;
		// look for another character to set the stride
		for next < len(chars) && !inCategory(next) {
			next++
		}
		if next >= len(chars) {
			// no more characters
			fmt.Printf(format, lo, hi, stride);
			break;
		}
		// set stride
		stride = next - lo;
		// check for length of run. next points to first jump in stride
		for i := next; i < len(chars); i++ {
			if inCategory(i) == (((i-lo)%stride) == 0) {
				// accept
				if inCategory(i) {
					hi = i
				}
			} else {
				// no more characters in this run
				break
			}
		}
		fmt.Printf(format, lo, hi, stride);
		// next range: start looking where this range ends
		next = hi + 1;
	}
	fmt.Print("}\n\n");
}

func fullCategoryTest(list []string) {
	for _, name := range list {
		if _, ok := category[name]; !ok {
			die.Log("unknown category", name);
		}
		r, ok := unicode.Categories[name];
		if !ok {
			die.Log("unknown table", name);
		}
		if name == "letter" {
			verifyRange(name, letterOp, r);
		} else {
			verifyRange(
				name,
				func(code int) bool { return chars[code].category == name },
				r
			);
		}
	}
}

func verifyRange(name string, inCategory Op, table []unicode.Range) {
	for i, c := range chars {
		web := inCategory(i);
		pkg := unicode.Is(table, i);
		if web != pkg {
			fmt.Fprintf(os.Stderr, "%s: U+%04X: web=%t pkg=%t\n", name, i, web, pkg);
		}
	}
}

func parseScript(line string) {
	comment := strings.Index(line, "#");
	if comment >= 0 {
		line = line[0:comment]
	}
	line = strings.TrimSpaceASCII(line);
	if len(line) == 0 {
		return
	}
	field := strings.Split(line, ";", -1);
	if len(field) != 2 {
		die.Logf("%s: %d fields (expected 2)\n", line, len(field));
	}
	matches := scriptRe.MatchStrings(line);
	if len(matches) != 4 {
		die.Logf("%s: %d matches (expected 3)\n", line, len(matches));
	}
	lo, err := strconv.Btoui64(matches[1], 16);
	if err != nil {
		die.Log("%.5s...:", err)
	}
	hi := lo;
	if len(matches[2]) > 2 {	// ignore leading ..
		hi, err = strconv.Btoui64(matches[2][2:len(matches[2])], 16);
		if err != nil {
			die.Log("%.5s...:", err)
		}
	}
	name := matches[3];
	s, ok := scripts[name];
	if len(s) == cap(s) {
		ns := make([]Script, len(s), len(s)+100);
		for i, sc := range s {
			ns[i] = sc
		}
		s = ns;
	}
	s = s[0:len(s)+1];
	s[len(s)-1] = Script{ uint32(lo), uint32(hi), name };
	scripts[name] = s;
}

func printScripts() {
	var err os.Error;
	scriptRe, err = regexp.Compile(`([0-9A-F]+)(\.\.[0-9A-F]+)? +; ([A-Za-z_]+)`);
	if err != nil {
		die.Log("re error:", err)
	}
	resp, _, err := http.Get(*url + "Scripts.txt");
	if err != nil {
		die.Log(err);
	}
	if resp.StatusCode != 200 {
		die.Log("bad GET status for Scripts.txt", resp.Status);
	}
	input := bufio.NewReader(resp.Body);
	for {
		line, err := input.ReadString('\n');
		if err != nil {
			if err == os.EOF {
				break;
			}
			die.Log(err);
		}
		parseScript(line[0:len(line)-1]);
	}
	resp.Body.Close();

	// Find out which scripts to dump
	list := strings.Split(*scriptlist, ",", 0);
	if *scriptlist == "all" {
		list = allScripts();
	}
	if *test {
		fullScriptTest(list);
		return;
	}

	fmt.Printf(
		"// Generated by running\n"
		"//	maketables --scripts=%s --url=%s\n"
		"// DO NOT EDIT\n\n",
		*scriptlist,
		*url
	);
	if *scriptlist == "all" {
		fmt.Println("// Scripts is the set of Unicode script tables.");
			fmt.Println("var Scripts = map[string] []Range {");
		for k, _ := range scripts {
			fmt.Printf("\t%q: %s,\n", k, k);
		}
		fmt.Printf("}\n\n");
	}

	decl := make(sort.StringArray, len(list));
	ndecl := 0;
	for _, name := range list {
		decl[ndecl] = fmt.Sprintf(
			"\t%s = _%s;\t// %s is the set of Unicode characters in script %s.\n",
			name, name, name, name
		);
		ndecl++;
		fmt.Printf("var _%s = []Range {\n", name);
		ranges := foldAdjacent(scripts[name]);
		for _, s := range ranges {
			fmt.Printf(format, s.Lo, s.Hi, s.Stride);
		}
		fmt.Printf("}\n\n");
	}
	decl.Sort();
	fmt.Println("var (");
	for _, d := range decl {
		fmt.Print(d);
	}
	fmt.Println(")\n");
}

// The script tables have a lot of adjacent elements. Fold them together.
func foldAdjacent(r []Script) []unicode.Range {
	s := make([]unicode.Range, 0, len(r));
	j := 0;
	for i := 0; i < len(r); i++ {
		if j>0 && int(r[i].lo) == s[j-1].Hi+1 {
			s[j-1].Hi = int(r[i].hi);
		} else {
			s = s[0:j+1];
			s[j] = unicode.Range{int(r[i].lo), int(r[i].hi), 1};
			j++;
		}
	}
	return s;
}

func fullScriptTest(list []string) {
	for _, name := range list {
		if _, ok := scripts[name]; !ok {
			die.Log("unknown script", name);
		}
		r, ok := unicode.Scripts[name];
		if !ok {
			die.Log("unknown table", name);
		}
		for _, script := range scripts[name] {
			for r := script.lo; r <= script.hi; r++ {
				if !unicode.Is(unicode.Scripts[name], int(r)) {
					fmt.Fprintf(os.Stderr, "U+%04X: not in script %s\n", r, name);
				}
			}
		}
	}
}
