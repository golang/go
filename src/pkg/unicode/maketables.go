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
	"strconv";
	"strings";
	"unicode";
)

var url = flag.String("url",
	"http://www.unicode.org/Public/5.1.0/ucd/UnicodeData.txt",
	"URL of Unicode database")
var tables = flag.String("tables",
	"all",
	"comma-separated list of which tables to generate; default is all; can be letter");
var test = flag.Bool("test",
	false,
	"test existing tables; can be used to compare web data with package data");

var die = log.New(os.Stderr, nil, "", log.Lexit|log.Lshortfile);

var category = map[string] bool{ "letter":true }	// Nd Lu etc. letter is a special case

// Data has form:
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

var chars = make([]Char, MaxChar)

var lastChar uint32 = 0;

func parse(line string) {
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

func main() {
	flag.Parse();

	resp, _, err := http.Get(*url);
	if err != nil {
		die.Log(err);
	}
	if resp.StatusCode != 200 {
		die.Log("bad GET status", resp.StatusCode);
	}
	input := bufio.NewReader(resp.Body);
	for {
		line, err := input.ReadLineString('\n', false);
		if err != nil {
			if err == os.EOF {
				break;
			}
			die.Log(err);
		}
		parse(line);
	}
	resp.Body.Close();
	// Find out which categories to dump
	list := strings.Split(*tables, ",", 0);
	if *tables == "all" {
		list = allCategories();
	}
	if *test {
		fullTest(list);
		return
	}
	fmt.Printf(
		"// Generated by running\n"
		"//	maketables --tables=%s --url=%s\n"
		"// DO NOT EDIT\n\n"
		"package unicode\n\n",
		*tables,
		*url
	);

	fmt.Println("// Version is the Unicode edition from which the tables are derived.");
	fmt.Printf("const Version = %q\n\n", version());

	if *tables == "all" {
		fmt.Println("// Tables is the set of Unicode data tables.");
			fmt.Println("var Tables = map[string] []Range {");
		for k, _ := range category {
			fmt.Printf("\t%q: %s,\n", k, k);
		}
		fmt.Printf("}\n\n");
	}

	for _, name := range list {
		if _, ok := category[name]; !ok {
			die.Log("unknown category", name);
		}
		// We generate an UpperCase name to serve as concise documentation and an _UnderScored
		// name to store the data.  This stops godoc dumping all the tables but keeps them
		// available to clients.
		if name == "letter" {	// special case
			dumpRange(
				"\n// Letter is the set of Unicode letters.\n"
				"var Letter = letter\n"
				"var letter = []Range {\n",
				letterOp,
				"}\n"
			);
			continue;
		}
		// Cases deserving special comments
		switch name {
		case "Nd":
			fmt.Printf(
				"\n// Digit is the set of Unicode characters with the \"decimal digit\" property.\n"
				"var Digit = Nd\n\n"
			)
		case "Lu":
			fmt.Printf(
				"\n// Upper is the set of Unicode upper case letters.\n"
				"var Upper = Lu\n\n"
			)
		case "Ll":
			fmt.Printf(
				"\n// Lower is the set of Unicode lower case letters.\n"
				"var Lower = Ll\n\n"
			)
		case "Lt":
			fmt.Printf(
				"\n// Title is the set of Unicode title case letters.\n"
				"var Title = Lt\n\n"
			)
		}
		dumpRange(
			fmt.Sprintf(
				"// %s is the set of Unicode characters in category %s\n"
				"var %s = _%s\n"
				"var _%s = []Range {\n",
				name, name, name, name, name
			),
			func(code int) bool { return chars[code].category == name },
			"}\n\n"
		);
	}
}

type Op func(code int) bool

func dumpRange(header string, inCategory Op, trailer string) {
	fmt.Print(header);
	const format = "\tRange{0x%04x, 0x%04x, %d},\n";
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
	fmt.Print(trailer);
}

func fullTest(list []string) {
	for _, name := range list {
		if _, ok := category[name]; !ok {
			die.Log("unknown category", name);
		}
		r, ok := unicode.Tables[name];
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
