// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Normalization table generator.
// Data read from the web.

package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"http"
	"io"
	"log"
	"os"
	"regexp"
	"strconv"
	"strings"
)

func main() {
	flag.Parse()
	loadUnicodeData()
	loadCompositionExclusions()
	completeCharFields(FCanonical)
	completeCharFields(FCompatibility)
	verifyComputed()
	printChars()
	makeTables()
	testDerived()
}

var url = flag.String("url",
	"http://www.unicode.org/Public/6.0.0/ucd/",
	"URL of Unicode database directory")
var tablelist = flag.String("tables",
	"all",
	"comma-separated list of which tables to generate; "+
		"can be 'decomp', 'recomp', 'info' and 'all'")
var test = flag.Bool("test",
	false,
	"test existing tables; can be used to compare web data with package data")
var verbose = flag.Bool("verbose",
	false,
	"write data to stdout as it is parsed")
var localFiles = flag.Bool("local",
	false,
	"data files have been copied to the current directory; for debugging only")

var logger = log.New(os.Stderr, "", log.Lshortfile)

// UnicodeData.txt has form:
//	0037;DIGIT SEVEN;Nd;0;EN;;7;7;7;N;;;;;
//	007A;LATIN SMALL LETTER Z;Ll;0;L;;;;;N;;;005A;;005A
// See http://unicode.org/reports/tr44/ for full explanation
// The fields:
const (
	FCodePoint = iota
	FName
	FGeneralCategory
	FCanonicalCombiningClass
	FBidiClass
	FDecompMapping
	FDecimalValue
	FDigitValue
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

// Quick Check properties of runes allow us to quickly
// determine whether a rune may occur in a normal form.
// For a given normal form, a rune may be guaranteed to occur
// verbatim (QC=Yes), may or may not combine with another 
// rune (QC=Maybe), or may not occur (QC=No).
type QCResult int

const (
	QCUnknown QCResult = iota
	QCYes
	QCNo
	QCMaybe
)

func (r QCResult) String() string {
	switch r {
	case QCYes:
		return "Yes"
	case QCNo:
		return "No"
	case QCMaybe:
		return "Maybe"
	}
	return "***UNKNOWN***"
}

const (
	FCanonical     = iota // NFC or NFD
	FCompatibility        // NFKC or NFKD
	FNumberOfFormTypes
)

const (
	MComposed   = iota // NFC or NFKC
	MDecomposed        // NFD or NFKD
	MNumberOfModes
)

// This contains only the properties we're interested in.
type Char struct {
	name          string
	codePoint     rune  // if zero, this index is not a valid code point.
	ccc           uint8 // canonical combining class
	excludeInComp bool  // from CompositionExclusions.txt
	compatDecomp  bool  // it has a compatibility expansion

	forms [FNumberOfFormTypes]FormInfo // For FCanonical and FCompatibility

	state State
}

var chars = make([]Char, MaxChar+1)

func (c Char) String() string {
	buf := new(bytes.Buffer)

	fmt.Fprintf(buf, "%U [%s]:\n", c.codePoint, c.name)
	fmt.Fprintf(buf, "  ccc: %v\n", c.ccc)
	fmt.Fprintf(buf, "  excludeInComp: %v\n", c.excludeInComp)
	fmt.Fprintf(buf, "  compatDecomp: %v\n", c.compatDecomp)
	fmt.Fprintf(buf, "  state: %v\n", c.state)
	fmt.Fprintf(buf, "  NFC:\n")
	fmt.Fprint(buf, c.forms[FCanonical])
	fmt.Fprintf(buf, "  NFKC:\n")
	fmt.Fprint(buf, c.forms[FCompatibility])

	return buf.String()
}

// In UnicodeData.txt, some ranges are marked like this:
//	3400;<CJK Ideograph Extension A, First>;Lo;0;L;;;;;N;;;;;
//	4DB5;<CJK Ideograph Extension A, Last>;Lo;0;L;;;;;N;;;;;
// parseCharacter keeps a state variable indicating the weirdness.
type State int

const (
	SNormal State = iota // known to be zero for the type
	SFirst
	SLast
	SMissing
)

var lastChar = rune('\u0000')

func (c Char) isValid() bool {
	return c.codePoint != 0 && c.state != SMissing
}

type FormInfo struct {
	quickCheck [MNumberOfModes]QCResult // index: MComposed or MDecomposed
	verified   [MNumberOfModes]bool     // index: MComposed or MDecomposed

	combinesForward  bool // May combine with rune on the right
	combinesBackward bool // May combine with rune on the left
	isOneWay         bool // Never appears in result
	inDecomp         bool // Some decompositions result in this char.
	decomp           Decomposition
	expandedDecomp   Decomposition
}

func (f FormInfo) String() string {
	buf := bytes.NewBuffer(make([]byte, 0))

	fmt.Fprintf(buf, "    quickCheck[C]: %v\n", f.quickCheck[MComposed])
	fmt.Fprintf(buf, "    quickCheck[D]: %v\n", f.quickCheck[MDecomposed])
	fmt.Fprintf(buf, "    cmbForward: %v\n", f.combinesForward)
	fmt.Fprintf(buf, "    cmbBackward: %v\n", f.combinesBackward)
	fmt.Fprintf(buf, "    isOneWay: %v\n", f.isOneWay)
	fmt.Fprintf(buf, "    inDecomp: %v\n", f.inDecomp)
	fmt.Fprintf(buf, "    decomposition: %v\n", f.decomp)
	fmt.Fprintf(buf, "    expandedDecomp: %v\n", f.expandedDecomp)

	return buf.String()
}

type Decomposition []rune

func (d Decomposition) String() string {
	return fmt.Sprintf("%.4X", d)
}

func openReader(file string) (input io.ReadCloser) {
	if *localFiles {
		f, err := os.Open(file)
		if err != nil {
			logger.Fatal(err)
		}
		input = f
	} else {
		path := *url + file
		resp, err := http.Get(path)
		if err != nil {
			logger.Fatal(err)
		}
		if resp.StatusCode != 200 {
			logger.Fatal("bad GET status for "+file, resp.Status)
		}
		input = resp.Body
	}
	return
}

func parseDecomposition(s string, skipfirst bool) (a []rune, e os.Error) {
	decomp := strings.Split(s, " ")
	if len(decomp) > 0 && skipfirst {
		decomp = decomp[1:]
	}
	for _, d := range decomp {
		point, err := strconv.Btoui64(d, 16)
		if err != nil {
			return a, err
		}
		a = append(a, rune(point))
	}
	return a, nil
}

func parseCharacter(line string) {
	field := strings.Split(line, ";")
	if len(field) != NumField {
		logger.Fatalf("%5s: %d fields (expected %d)\n", line, len(field), NumField)
	}
	x, err := strconv.Btoui64(field[FCodePoint], 16)
	point := int(x)
	if err != nil {
		logger.Fatalf("%.5s...: %s", line, err)
	}
	if point == 0 {
		return // not interesting and we use 0 as unset
	}
	if point > MaxChar {
		logger.Fatalf("%5s: Rune %X > MaxChar (%X)", line, point, MaxChar)
		return
	}
	state := SNormal
	switch {
	case strings.Index(field[FName], ", First>") > 0:
		state = SFirst
	case strings.Index(field[FName], ", Last>") > 0:
		state = SLast
	}
	firstChar := lastChar + 1
	lastChar = rune(point)
	if state != SLast {
		firstChar = lastChar
	}
	x, err = strconv.Atoui64(field[FCanonicalCombiningClass])
	if err != nil {
		logger.Fatalf("%U: bad ccc field: %s", int(x), err)
	}
	ccc := uint8(x)
	decmap := field[FDecompMapping]
	exp, e := parseDecomposition(decmap, false)
	isCompat := false
	if e != nil {
		if len(decmap) > 0 {
			exp, e = parseDecomposition(decmap, true)
			if e != nil {
				logger.Fatalf(`%U: bad decomp |%v|: "%s"`, int(x), decmap, e)
			}
			isCompat = true
		}
	}
	for i := firstChar; i <= lastChar; i++ {
		char := &chars[i]
		char.name = field[FName]
		char.codePoint = i
		char.forms[FCompatibility].decomp = exp
		if !isCompat {
			char.forms[FCanonical].decomp = exp
		} else {
			char.compatDecomp = true
		}
		if len(decmap) > 0 {
			char.forms[FCompatibility].decomp = exp
		}
		char.ccc = ccc
		char.state = SMissing
		if i == lastChar {
			char.state = state
		}
	}
	return
}

func loadUnicodeData() {
	f := openReader("UnicodeData.txt")
	defer f.Close()
	input := bufio.NewReader(f)
	for {
		line, err := input.ReadString('\n')
		if err != nil {
			if err == os.EOF {
				break
			}
			logger.Fatal(err)
		}
		parseCharacter(line[0 : len(line)-1])
	}
}

var singlePointRe = regexp.MustCompile(`^([0-9A-F]+) *$`)

// CompositionExclusions.txt has form:
// 0958    # ...
// See http://unicode.org/reports/tr44/ for full explanation
func parseExclusion(line string) int {
	comment := strings.Index(line, "#")
	if comment >= 0 {
		line = line[0:comment]
	}
	if len(line) == 0 {
		return 0
	}
	matches := singlePointRe.FindStringSubmatch(line)
	if len(matches) != 2 {
		logger.Fatalf("%s: %d matches (expected 1)\n", line, len(matches))
	}
	point, err := strconv.Btoui64(matches[1], 16)
	if err != nil {
		logger.Fatalf("%.5s...: %s", line, err)
	}
	return int(point)
}

func loadCompositionExclusions() {
	f := openReader("CompositionExclusions.txt")
	defer f.Close()
	input := bufio.NewReader(f)
	for {
		line, err := input.ReadString('\n')
		if err != nil {
			if err == os.EOF {
				break
			}
			logger.Fatal(err)
		}
		point := parseExclusion(line[0 : len(line)-1])
		if point == 0 {
			continue
		}
		c := &chars[point]
		if c.excludeInComp {
			logger.Fatalf("%U: Duplicate entry in exclusions.", c.codePoint)
		}
		c.excludeInComp = true
	}
}

// hasCompatDecomp returns true if any of the recursive
// decompositions contains a compatibility expansion.
// In this case, the character may not occur in NFK*.
func hasCompatDecomp(r rune) bool {
	c := &chars[r]
	if c.compatDecomp {
		return true
	}
	for _, d := range c.forms[FCompatibility].decomp {
		if hasCompatDecomp(d) {
			return true
		}
	}
	return false
}

// Hangul related constants.
const (
	HangulBase = 0xAC00
	HangulEnd  = 0xD7A4 // hangulBase + Jamo combinations (19 * 21 * 28)

	JamoLBase = 0x1100
	JamoLEnd  = 0x1113
	JamoVBase = 0x1161
	JamoVEnd  = 0x1176
	JamoTBase = 0x11A8
	JamoTEnd  = 0x11C3
)

func isHangul(r rune) bool {
	return HangulBase <= r && r < HangulEnd
}

func ccc(r rune) uint8 {
	return chars[r].ccc
}

// Insert a rune in a buffer, ordered by Canonical Combining Class.
func insertOrdered(b Decomposition, r rune) Decomposition {
	n := len(b)
	b = append(b, 0)
	cc := ccc(r)
	if cc > 0 {
		// Use bubble sort.
		for ; n > 0; n-- {
			if ccc(b[n-1]) <= cc {
				break
			}
			b[n] = b[n-1]
		}
	}
	b[n] = r
	return b
}

// Recursively decompose.
func decomposeRecursive(form int, r rune, d Decomposition) Decomposition {
	if isHangul(r) {
		return d
	}
	dcomp := chars[r].forms[form].decomp
	if len(dcomp) == 0 {
		return insertOrdered(d, r)
	}
	for _, c := range dcomp {
		d = decomposeRecursive(form, c, d)
	}
	return d
}

func completeCharFields(form int) {
	// Phase 0: pre-expand decomposition.
	for i := range chars {
		f := &chars[i].forms[form]
		if len(f.decomp) == 0 {
			continue
		}
		exp := make(Decomposition, 0)
		for _, c := range f.decomp {
			exp = decomposeRecursive(form, c, exp)
		}
		f.expandedDecomp = exp
	}

	// Phase 1: composition exclusion, mark decomposition.
	for i := range chars {
		c := &chars[i]
		f := &c.forms[form]

		// Marks script-specific exclusions and version restricted.
		f.isOneWay = c.excludeInComp

		// Singletons
		f.isOneWay = f.isOneWay || len(f.decomp) == 1

		// Non-starter decompositions
		if len(f.decomp) > 1 {
			chk := c.ccc != 0 || chars[f.decomp[0]].ccc != 0
			f.isOneWay = f.isOneWay || chk
		}

		// Runes that decompose into more than two runes.
		f.isOneWay = f.isOneWay || len(f.decomp) > 2

		if form == FCompatibility {
			f.isOneWay = f.isOneWay || hasCompatDecomp(c.codePoint)
		}

		for _, r := range f.decomp {
			chars[r].forms[form].inDecomp = true
		}
	}

	// Phase 2: forward and backward combining.
	for i := range chars {
		c := &chars[i]
		f := &c.forms[form]

		if !f.isOneWay && len(f.decomp) == 2 {
			f0 := &chars[f.decomp[0]].forms[form]
			f1 := &chars[f.decomp[1]].forms[form]
			if !f0.isOneWay {
				f0.combinesForward = true
			}
			if !f1.isOneWay {
				f1.combinesBackward = true
			}
		}
	}

	// Phase 3: quick check values.
	for i := range chars {
		c := &chars[i]
		f := &c.forms[form]

		switch {
		case len(f.decomp) > 0:
			f.quickCheck[MDecomposed] = QCNo
		case isHangul(rune(i)):
			f.quickCheck[MDecomposed] = QCNo
		default:
			f.quickCheck[MDecomposed] = QCYes
		}
		switch {
		case f.isOneWay:
			f.quickCheck[MComposed] = QCNo
		case (i & 0xffff00) == JamoLBase:
			f.quickCheck[MComposed] = QCYes
			if JamoLBase <= i && i < JamoLEnd {
				f.combinesForward = true
			}
			if JamoVBase <= i && i < JamoVEnd {
				f.quickCheck[MComposed] = QCMaybe
				f.combinesBackward = true
				f.combinesForward = true
			}
			if JamoTBase <= i && i < JamoTEnd {
				f.quickCheck[MComposed] = QCMaybe
				f.combinesBackward = true
			}
		case !f.combinesBackward:
			f.quickCheck[MComposed] = QCYes
		default:
			f.quickCheck[MComposed] = QCMaybe
		}
	}
}

func printBytes(b []byte, name string) {
	fmt.Printf("// %s: %d bytes\n", name, len(b))
	fmt.Printf("var %s = [...]byte {", name)
	for i, c := range b {
		switch {
		case i%64 == 0:
			fmt.Printf("\n// Bytes %x - %x\n", i, i+63)
		case i%8 == 0:
			fmt.Printf("\n")
		}
		fmt.Printf("0x%.2X, ", c)
	}
	fmt.Print("\n}\n\n")
}

// See forminfo.go for format.
func makeEntry(f *FormInfo) uint16 {
	e := uint16(0)
	if f.combinesForward {
		e |= 0x8
	}
	if f.quickCheck[MDecomposed] == QCNo {
		e |= 0x1
	}
	switch f.quickCheck[MComposed] {
	case QCYes:
	case QCNo:
		e |= 0x2
	case QCMaybe:
		e |= 0x6
	default:
		log.Fatalf("Illegal quickcheck value %v.", f.quickCheck[MComposed])
	}
	return e
}

// Bits
// 0..8:   CCC
// 9..12:  NF(C|D) qc bits.
// 13..16: NFK(C|D) qc bits.
func makeCharInfo(c Char) uint16 {
	e := makeEntry(&c.forms[FCompatibility])
	e = e<<4 | makeEntry(&c.forms[FCanonical])
	e = e<<8 | uint16(c.ccc)
	return e
}

func printCharInfoTables() int {
	// Quick Check + CCC trie.
	t := newNode()
	for i, char := range chars {
		v := makeCharInfo(char)
		if v != 0 {
			t.insert(rune(i), v)
		}
	}
	return t.printTables("charInfo")
}

func printDecompositionTables() int {
	decompositions := bytes.NewBuffer(make([]byte, 0, 10000))
	size := 0

	// Map decompositions
	positionMap := make(map[string]uint16)

	// Store the uniqued decompositions in a byte buffer,
	// preceded by their byte length.
	for _, c := range chars {
		for f := 0; f < 2; f++ {
			d := c.forms[f].expandedDecomp
			s := string([]rune(d))
			if _, ok := positionMap[s]; !ok {
				p := decompositions.Len()
				decompositions.WriteByte(uint8(len(s)))
				decompositions.WriteString(s)
				positionMap[s] = uint16(p)
			}
		}
	}
	b := decompositions.Bytes()
	printBytes(b, "decomps")
	size += len(b)

	nfcT := newNode()
	nfkcT := newNode()
	for i, c := range chars {
		d := c.forms[FCanonical].expandedDecomp
		if len(d) != 0 {
			nfcT.insert(rune(i), positionMap[string([]rune(d))])
			if ccc(c.codePoint) != ccc(d[0]) {
				// We assume the lead ccc of a decomposition is !=0 in this case.
				if ccc(d[0]) == 0 {
					logger.Fatal("Expected differing CCC to be non-zero.")
				}
			}
		}
		d = c.forms[FCompatibility].expandedDecomp
		if len(d) != 0 {
			nfkcT.insert(rune(i), positionMap[string([]rune(d))])
			if ccc(c.codePoint) != ccc(d[0]) {
				// We assume the lead ccc of a decomposition is !=0 in this case.
				if ccc(d[0]) == 0 {
					logger.Fatal("Expected differing CCC to be non-zero.")
				}
			}
		}
	}
	size += nfcT.printTables("nfcDecomp")
	size += nfkcT.printTables("nfkcDecomp")
	return size
}

func contains(sa []string, s string) bool {
	for _, a := range sa {
		if a == s {
			return true
		}
	}
	return false
}

// Extract the version number from the URL.
func version() string {
	// From http://www.unicode.org/standard/versions/#Version_Numbering:
	// for the later Unicode versions, data files are located in
	// versioned directories.
	fields := strings.Split(*url, "/")
	for _, f := range fields {
		if match, _ := regexp.MatchString(`[0-9]\.[0-9]\.[0-9]`, f); match {
			return f
		}
	}
	logger.Fatal("unknown version")
	return "Unknown"
}

const fileHeader = `// Generated by running
//	maketables --tables=%s --url=%s
// DO NOT EDIT

package norm

`

func makeTables() {
	size := 0
	if *tablelist == "" {
		return
	}
	list := strings.Split(*tablelist, ",")
	if *tablelist == "all" {
		list = []string{"decomp", "recomp", "info"}
	}
	fmt.Printf(fileHeader, *tablelist, *url)

	fmt.Println("// Version is the Unicode edition from which the tables are derived.")
	fmt.Printf("const Version = %q\n\n", version())

	if contains(list, "decomp") {
		size += printDecompositionTables()
	}

	if contains(list, "recomp") {
		// Note that we use 32 bit keys, instead of 64 bit.
		// This clips the bits of three entries, but we know
		// this won't cause a collision. The compiler will catch
		// any changes made to UnicodeData.txt that introduces
		// a collision.
		// Note that the recomposition map for NFC and NFKC
		// are identical.

		// Recomposition map
		nrentries := 0
		for _, c := range chars {
			f := c.forms[FCanonical]
			if !f.isOneWay && len(f.decomp) > 0 {
				nrentries++
			}
		}
		sz := nrentries * 8
		size += sz
		fmt.Printf("// recompMap: %d bytes (entries only)\n", sz)
		fmt.Println("var recompMap = map[uint32]uint32{")
		for i, c := range chars {
			f := c.forms[FCanonical]
			d := f.decomp
			if !f.isOneWay && len(d) > 0 {
				key := uint32(uint16(d[0]))<<16 + uint32(uint16(d[1]))
				fmt.Printf("0x%.8X: 0x%.4X,\n", key, i)
			}
		}
		fmt.Printf("}\n\n")
	}

	if contains(list, "info") {
		size += printCharInfoTables()
	}
	fmt.Printf("// Total size of tables: %dKB (%d bytes)\n", (size+512)/1024, size)
}

func printChars() {
	if *verbose {
		for _, c := range chars {
			if !c.isValid() || c.state == SMissing {
				continue
			}
			fmt.Println(c)
		}
	}
}

// verifyComputed does various consistency tests.
func verifyComputed() {
	for i, c := range chars {
		for _, f := range c.forms {
			isNo := (f.quickCheck[MDecomposed] == QCNo)
			if (len(f.decomp) > 0) != isNo && !isHangul(rune(i)) {
				log.Fatalf("%U: NF*D must be no if rune decomposes", i)
			}

			isMaybe := f.quickCheck[MComposed] == QCMaybe
			if f.combinesBackward != isMaybe {
				log.Fatalf("%U: NF*C must be maybe if combinesBackward", i)
			}
		}
	}
}

var qcRe = regexp.MustCompile(`([0-9A-F\.]+) *; (NF.*_QC); ([YNM]) #.*`)

// Use values in DerivedNormalizationProps.txt to compare against the
// values we computed.
// DerivedNormalizationProps.txt has form:
// 00C0..00C5    ; NFD_QC; N # ...
// 0374          ; NFD_QC; N # ...
// See http://unicode.org/reports/tr44/ for full explanation
func testDerived() {
	if !*test {
		return
	}
	f := openReader("DerivedNormalizationProps.txt")
	defer f.Close()
	input := bufio.NewReader(f)
	for {
		line, err := input.ReadString('\n')
		if err != nil {
			if err == os.EOF {
				break
			}
			logger.Fatal(err)
		}
		qc := qcRe.FindStringSubmatch(line)
		if qc == nil {
			continue
		}
		rng := strings.Split(qc[1], "..")
		i, err := strconv.Btoui64(rng[0], 16)
		if err != nil {
			log.Fatal(err)
		}
		j := i
		if len(rng) > 1 {
			j, err = strconv.Btoui64(rng[1], 16)
			if err != nil {
				log.Fatal(err)
			}
		}
		var ftype, mode int
		qt := strings.TrimSpace(qc[2])
		switch qt {
		case "NFC_QC":
			ftype, mode = FCanonical, MComposed
		case "NFD_QC":
			ftype, mode = FCanonical, MDecomposed
		case "NFKC_QC":
			ftype, mode = FCompatibility, MComposed
		case "NFKD_QC":
			ftype, mode = FCompatibility, MDecomposed
		default:
			log.Fatalf(`Unexpected quick check type "%s"`, qt)
		}
		var qr QCResult
		switch qc[3] {
		case "Y":
			qr = QCYes
		case "N":
			qr = QCNo
		case "M":
			qr = QCMaybe
		default:
			log.Fatalf(`Unexpected quick check value "%s"`, qc[3])
		}
		var lastFailed bool
		// Verify current
		for ; i <= j; i++ {
			c := &chars[int(i)]
			c.forms[ftype].verified[mode] = true
			curqr := c.forms[ftype].quickCheck[mode]
			if curqr != qr {
				if !lastFailed {
					logger.Printf("%s: %.4X..%.4X -- %s\n",
						qt, int(i), int(j), line[0:50])
				}
				logger.Printf("%U: FAILED %s (was %v need %v)\n",
					int(i), qt, curqr, qr)
				lastFailed = true
			}
		}
	}
	// Any unspecified value must be QCYes. Verify this.
	for i, c := range chars {
		for j, fd := range c.forms {
			for k, qr := range fd.quickCheck {
				if !fd.verified[k] && qr != QCYes {
					m := "%U: FAIL F:%d M:%d (was %v need Yes) %s\n"
					logger.Printf(m, i, j, k, qr, c.name)
				}
			}
		}
	}
}
