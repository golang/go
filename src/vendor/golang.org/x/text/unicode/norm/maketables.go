// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// Normalization table generator.
// Data read from the web.
// See forminfo.go for a description of the trie values associated with each rune.

package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"log"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/triegen"
	"golang.org/x/text/internal/ucd"
)

func main() {
	gen.Init()
	loadUnicodeData()
	compactCCC()
	loadCompositionExclusions()
	completeCharFields(FCanonical)
	completeCharFields(FCompatibility)
	computeNonStarterCounts()
	verifyComputed()
	printChars()
	testDerived()
	printTestdata()
	makeTables()
}

var (
	tablelist = flag.String("tables",
		"all",
		"comma-separated list of which tables to generate; "+
			"can be 'decomp', 'recomp', 'info' and 'all'")
	test = flag.Bool("test",
		false,
		"test existing tables against DerivedNormalizationProps and generate test data for regression testing")
	verbose = flag.Bool("verbose",
		false,
		"write data to stdout as it is parsed")
)

const MaxChar = 0x10FFFF // anything above this shouldn't exist

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
	origCCC       uint8
	excludeInComp bool // from CompositionExclusions.txt
	compatDecomp  bool // it has a compatibility expansion

	nTrailingNonStarters uint8
	nLeadingNonStarters  uint8 // must be equal to trailing if non-zero

	forms [FNumberOfFormTypes]FormInfo // For FCanonical and FCompatibility

	state State
}

var chars = make([]Char, MaxChar+1)
var cccMap = make(map[uint8]uint8)

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
	fmt.Fprintf(buf, "    decomposition: %X\n", f.decomp)
	fmt.Fprintf(buf, "    expandedDecomp: %X\n", f.expandedDecomp)

	return buf.String()
}

type Decomposition []rune

func parseDecomposition(s string, skipfirst bool) (a []rune, err error) {
	decomp := strings.Split(s, " ")
	if len(decomp) > 0 && skipfirst {
		decomp = decomp[1:]
	}
	for _, d := range decomp {
		point, err := strconv.ParseUint(d, 16, 64)
		if err != nil {
			return a, err
		}
		a = append(a, rune(point))
	}
	return a, nil
}

func loadUnicodeData() {
	f := gen.OpenUCDFile("UnicodeData.txt")
	defer f.Close()
	p := ucd.New(f)
	for p.Next() {
		r := p.Rune(ucd.CodePoint)
		char := &chars[r]

		char.ccc = uint8(p.Uint(ucd.CanonicalCombiningClass))
		decmap := p.String(ucd.DecompMapping)

		exp, err := parseDecomposition(decmap, false)
		isCompat := false
		if err != nil {
			if len(decmap) > 0 {
				exp, err = parseDecomposition(decmap, true)
				if err != nil {
					log.Fatalf(`%U: bad decomp |%v|: "%s"`, r, decmap, err)
				}
				isCompat = true
			}
		}

		char.name = p.String(ucd.Name)
		char.codePoint = r
		char.forms[FCompatibility].decomp = exp
		if !isCompat {
			char.forms[FCanonical].decomp = exp
		} else {
			char.compatDecomp = true
		}
		if len(decmap) > 0 {
			char.forms[FCompatibility].decomp = exp
		}
	}
	if err := p.Err(); err != nil {
		log.Fatal(err)
	}
}

// compactCCC converts the sparse set of CCC values to a continguous one,
// reducing the number of bits needed from 8 to 6.
func compactCCC() {
	m := make(map[uint8]uint8)
	for i := range chars {
		c := &chars[i]
		m[c.ccc] = 0
	}
	cccs := []int{}
	for v, _ := range m {
		cccs = append(cccs, int(v))
	}
	sort.Ints(cccs)
	for i, c := range cccs {
		cccMap[uint8(i)] = uint8(c)
		m[uint8(c)] = uint8(i)
	}
	for i := range chars {
		c := &chars[i]
		c.origCCC = c.ccc
		c.ccc = m[c.ccc]
	}
	if len(m) >= 1<<6 {
		log.Fatalf("too many difference CCC values: %d >= 64", len(m))
	}
}

// CompositionExclusions.txt has form:
// 0958    # ...
// See http://unicode.org/reports/tr44/ for full explanation
func loadCompositionExclusions() {
	f := gen.OpenUCDFile("CompositionExclusions.txt")
	defer f.Close()
	p := ucd.New(f)
	for p.Next() {
		c := &chars[p.Rune(0)]
		if c.excludeInComp {
			log.Fatalf("%U: Duplicate entry in exclusions.", c.codePoint)
		}
		c.excludeInComp = true
	}
	if e := p.Err(); e != nil {
		log.Fatal(e)
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

	JamoLVTCount = 19 * 21 * 28
	JamoTCount   = 28
)

func isHangul(r rune) bool {
	return HangulBase <= r && r < HangulEnd
}

func isHangulWithoutJamoT(r rune) bool {
	if !isHangul(r) {
		return false
	}
	r -= HangulBase
	return r < JamoLVTCount && r%JamoTCount == 0
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
		if isHangulWithoutJamoT(rune(i)) {
			f.combinesForward = true
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

func computeNonStarterCounts() {
	// Phase 4: leading and trailing non-starter count
	for i := range chars {
		c := &chars[i]

		runes := []rune{rune(i)}
		// We always use FCompatibility so that the CGJ insertion points do not
		// change for repeated normalizations with different forms.
		if exp := c.forms[FCompatibility].expandedDecomp; len(exp) > 0 {
			runes = exp
		}
		// We consider runes that combine backwards to be non-starters for the
		// purpose of Stream-Safe Text Processing.
		for _, r := range runes {
			if cr := &chars[r]; cr.ccc == 0 && !cr.forms[FCompatibility].combinesBackward {
				break
			}
			c.nLeadingNonStarters++
		}
		for i := len(runes) - 1; i >= 0; i-- {
			if cr := &chars[runes[i]]; cr.ccc == 0 && !cr.forms[FCompatibility].combinesBackward {
				break
			}
			c.nTrailingNonStarters++
		}
		if c.nTrailingNonStarters > 3 {
			log.Fatalf("%U: Decomposition with more than 3 (%d) trailing modifiers (%U)", i, c.nTrailingNonStarters, runes)
		}

		if isHangul(rune(i)) {
			c.nTrailingNonStarters = 2
			if isHangulWithoutJamoT(rune(i)) {
				c.nTrailingNonStarters = 1
			}
		}

		if l, t := c.nLeadingNonStarters, c.nTrailingNonStarters; l > 0 && l != t {
			log.Fatalf("%U: number of leading and trailing non-starters should be equal (%d vs %d)", i, l, t)
		}
		if t := c.nTrailingNonStarters; t > 3 {
			log.Fatalf("%U: number of trailing non-starters is %d > 3", t)
		}
	}
}

func printBytes(w io.Writer, b []byte, name string) {
	fmt.Fprintf(w, "// %s: %d bytes\n", name, len(b))
	fmt.Fprintf(w, "var %s = [...]byte {", name)
	for i, c := range b {
		switch {
		case i%64 == 0:
			fmt.Fprintf(w, "\n// Bytes %x - %x\n", i, i+63)
		case i%8 == 0:
			fmt.Fprintf(w, "\n")
		}
		fmt.Fprintf(w, "0x%.2X, ", c)
	}
	fmt.Fprint(w, "\n}\n\n")
}

// See forminfo.go for format.
func makeEntry(f *FormInfo, c *Char) uint16 {
	e := uint16(0)
	if r := c.codePoint; HangulBase <= r && r < HangulEnd {
		e |= 0x40
	}
	if f.combinesForward {
		e |= 0x20
	}
	if f.quickCheck[MDecomposed] == QCNo {
		e |= 0x4
	}
	switch f.quickCheck[MComposed] {
	case QCYes:
	case QCNo:
		e |= 0x10
	case QCMaybe:
		e |= 0x18
	default:
		log.Fatalf("Illegal quickcheck value %v.", f.quickCheck[MComposed])
	}
	e |= uint16(c.nTrailingNonStarters)
	return e
}

// decompSet keeps track of unique decompositions, grouped by whether
// the decomposition is followed by a trailing and/or leading CCC.
type decompSet [7]map[string]bool

const (
	normalDecomp = iota
	firstMulti
	firstCCC
	endMulti
	firstLeadingCCC
	firstCCCZeroExcept
	firstStarterWithNLead
	lastDecomp
)

var cname = []string{"firstMulti", "firstCCC", "endMulti", "firstLeadingCCC", "firstCCCZeroExcept", "firstStarterWithNLead", "lastDecomp"}

func makeDecompSet() decompSet {
	m := decompSet{}
	for i := range m {
		m[i] = make(map[string]bool)
	}
	return m
}
func (m *decompSet) insert(key int, s string) {
	m[key][s] = true
}

func printCharInfoTables(w io.Writer) int {
	mkstr := func(r rune, f *FormInfo) (int, string) {
		d := f.expandedDecomp
		s := string([]rune(d))
		if max := 1 << 6; len(s) >= max {
			const msg = "%U: too many bytes in decomposition: %d >= %d"
			log.Fatalf(msg, r, len(s), max)
		}
		head := uint8(len(s))
		if f.quickCheck[MComposed] != QCYes {
			head |= 0x40
		}
		if f.combinesForward {
			head |= 0x80
		}
		s = string([]byte{head}) + s

		lccc := ccc(d[0])
		tccc := ccc(d[len(d)-1])
		cc := ccc(r)
		if cc != 0 && lccc == 0 && tccc == 0 {
			log.Fatalf("%U: trailing and leading ccc are 0 for non-zero ccc %d", r, cc)
		}
		if tccc < lccc && lccc != 0 {
			const msg = "%U: lccc (%d) must be <= tcc (%d)"
			log.Fatalf(msg, r, lccc, tccc)
		}
		index := normalDecomp
		nTrail := chars[r].nTrailingNonStarters
		nLead := chars[r].nLeadingNonStarters
		if tccc > 0 || lccc > 0 || nTrail > 0 {
			tccc <<= 2
			tccc |= nTrail
			s += string([]byte{tccc})
			index = endMulti
			for _, r := range d[1:] {
				if ccc(r) == 0 {
					index = firstCCC
				}
			}
			if lccc > 0 || nLead > 0 {
				s += string([]byte{lccc})
				if index == firstCCC {
					log.Fatalf("%U: multi-segment decomposition not supported for decompositions with leading CCC != 0", r)
				}
				index = firstLeadingCCC
			}
			if cc != lccc {
				if cc != 0 {
					log.Fatalf("%U: for lccc != ccc, expected ccc to be 0; was %d", r, cc)
				}
				index = firstCCCZeroExcept
			}
		} else if len(d) > 1 {
			index = firstMulti
		}
		return index, s
	}

	decompSet := makeDecompSet()
	const nLeadStr = "\x00\x01" // 0-byte length and tccc with nTrail.
	decompSet.insert(firstStarterWithNLead, nLeadStr)

	// Store the uniqued decompositions in a byte buffer,
	// preceded by their byte length.
	for _, c := range chars {
		for _, f := range c.forms {
			if len(f.expandedDecomp) == 0 {
				continue
			}
			if f.combinesBackward {
				log.Fatalf("%U: combinesBackward and decompose", c.codePoint)
			}
			index, s := mkstr(c.codePoint, &f)
			decompSet.insert(index, s)
		}
	}

	decompositions := bytes.NewBuffer(make([]byte, 0, 10000))
	size := 0
	positionMap := make(map[string]uint16)
	decompositions.WriteString("\000")
	fmt.Fprintln(w, "const (")
	for i, m := range decompSet {
		sa := []string{}
		for s := range m {
			sa = append(sa, s)
		}
		sort.Strings(sa)
		for _, s := range sa {
			p := decompositions.Len()
			decompositions.WriteString(s)
			positionMap[s] = uint16(p)
		}
		if cname[i] != "" {
			fmt.Fprintf(w, "%s = 0x%X\n", cname[i], decompositions.Len())
		}
	}
	fmt.Fprintln(w, "maxDecomp = 0x8000")
	fmt.Fprintln(w, ")")
	b := decompositions.Bytes()
	printBytes(w, b, "decomps")
	size += len(b)

	varnames := []string{"nfc", "nfkc"}
	for i := 0; i < FNumberOfFormTypes; i++ {
		trie := triegen.NewTrie(varnames[i])

		for r, c := range chars {
			f := c.forms[i]
			d := f.expandedDecomp
			if len(d) != 0 {
				_, key := mkstr(c.codePoint, &f)
				trie.Insert(rune(r), uint64(positionMap[key]))
				if c.ccc != ccc(d[0]) {
					// We assume the lead ccc of a decomposition !=0 in this case.
					if ccc(d[0]) == 0 {
						log.Fatalf("Expected leading CCC to be non-zero; ccc is %d", c.ccc)
					}
				}
			} else if c.nLeadingNonStarters > 0 && len(f.expandedDecomp) == 0 && c.ccc == 0 && !f.combinesBackward {
				// Handle cases where it can't be detected that the nLead should be equal
				// to nTrail.
				trie.Insert(c.codePoint, uint64(positionMap[nLeadStr]))
			} else if v := makeEntry(&f, &c)<<8 | uint16(c.ccc); v != 0 {
				trie.Insert(c.codePoint, uint64(0x8000|v))
			}
		}
		sz, err := trie.Gen(w, triegen.Compact(&normCompacter{name: varnames[i]}))
		if err != nil {
			log.Fatal(err)
		}
		size += sz
	}
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

func makeTables() {
	w := &bytes.Buffer{}

	size := 0
	if *tablelist == "" {
		return
	}
	list := strings.Split(*tablelist, ",")
	if *tablelist == "all" {
		list = []string{"recomp", "info"}
	}

	// Compute maximum decomposition size.
	max := 0
	for _, c := range chars {
		if n := len(string(c.forms[FCompatibility].expandedDecomp)); n > max {
			max = n
		}
	}

	fmt.Fprintln(w, "const (")
	fmt.Fprintln(w, "\t// Version is the Unicode edition from which the tables are derived.")
	fmt.Fprintf(w, "\tVersion = %q\n", gen.UnicodeVersion())
	fmt.Fprintln(w)
	fmt.Fprintln(w, "\t// MaxTransformChunkSize indicates the maximum number of bytes that Transform")
	fmt.Fprintln(w, "\t// may need to write atomically for any Form. Making a destination buffer at")
	fmt.Fprintln(w, "\t// least this size ensures that Transform can always make progress and that")
	fmt.Fprintln(w, "\t// the user does not need to grow the buffer on an ErrShortDst.")
	fmt.Fprintf(w, "\tMaxTransformChunkSize = %d+maxNonStarters*4\n", len(string(0x034F))+max)
	fmt.Fprintln(w, ")\n")

	// Print the CCC remap table.
	size += len(cccMap)
	fmt.Fprintf(w, "var ccc = [%d]uint8{", len(cccMap))
	for i := 0; i < len(cccMap); i++ {
		if i%8 == 0 {
			fmt.Fprintln(w)
		}
		fmt.Fprintf(w, "%3d, ", cccMap[uint8(i)])
	}
	fmt.Fprintln(w, "\n}\n")

	if contains(list, "info") {
		size += printCharInfoTables(w)
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
		fmt.Fprintf(w, "// recompMap: %d bytes (entries only)\n", sz)
		fmt.Fprintln(w, "var recompMap = map[uint32]rune{")
		for i, c := range chars {
			f := c.forms[FCanonical]
			d := f.decomp
			if !f.isOneWay && len(d) > 0 {
				key := uint32(uint16(d[0]))<<16 + uint32(uint16(d[1]))
				fmt.Fprintf(w, "0x%.8X: 0x%.4X,\n", key, i)
			}
		}
		fmt.Fprintf(w, "}\n\n")
	}

	fmt.Fprintf(w, "// Total size of tables: %dKB (%d bytes)\n", (size+512)/1024, size)
	gen.WriteGoFile("tables.go", "norm", w.Bytes())
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
				log.Fatalf("%U: NF*D QC must be No if rune decomposes", i)
			}

			isMaybe := f.quickCheck[MComposed] == QCMaybe
			if f.combinesBackward != isMaybe {
				log.Fatalf("%U: NF*C QC must be Maybe if combinesBackward", i)
			}
			if len(f.decomp) > 0 && f.combinesForward && isMaybe {
				log.Fatalf("%U: NF*C QC must be Yes or No if combinesForward and decomposes", i)
			}

			if len(f.expandedDecomp) != 0 {
				continue
			}
			if a, b := c.nLeadingNonStarters > 0, (c.ccc > 0 || f.combinesBackward); a != b {
				// We accept these runes to be treated differently (it only affects
				// segment breaking in iteration, most likely on improper use), but
				// reconsider if more characters are added.
				// U+FF9E HALFWIDTH KATAKANA VOICED SOUND MARK;Lm;0;L;<narrow> 3099;;;;N;;;;;
				// U+FF9F HALFWIDTH KATAKANA SEMI-VOICED SOUND MARK;Lm;0;L;<narrow> 309A;;;;N;;;;;
				// U+3133 HANGUL LETTER KIYEOK-SIOS;Lo;0;L;<compat> 11AA;;;;N;HANGUL LETTER GIYEOG SIOS;;;;
				// U+318E HANGUL LETTER ARAEAE;Lo;0;L;<compat> 11A1;;;;N;HANGUL LETTER ALAE AE;;;;
				// U+FFA3 HALFWIDTH HANGUL LETTER KIYEOK-SIOS;Lo;0;L;<narrow> 3133;;;;N;HALFWIDTH HANGUL LETTER GIYEOG SIOS;;;;
				// U+FFDC HALFWIDTH HANGUL LETTER I;Lo;0;L;<narrow> 3163;;;;N;;;;;
				if i != 0xFF9E && i != 0xFF9F && !(0x3133 <= i && i <= 0x318E) && !(0xFFA3 <= i && i <= 0xFFDC) {
					log.Fatalf("%U: nLead was %v; want %v", i, a, b)
				}
			}
		}
		nfc := c.forms[FCanonical]
		nfkc := c.forms[FCompatibility]
		if nfc.combinesBackward != nfkc.combinesBackward {
			log.Fatalf("%U: Cannot combine combinesBackward\n", c.codePoint)
		}
	}
}

// Use values in DerivedNormalizationProps.txt to compare against the
// values we computed.
// DerivedNormalizationProps.txt has form:
// 00C0..00C5    ; NFD_QC; N # ...
// 0374          ; NFD_QC; N # ...
// See http://unicode.org/reports/tr44/ for full explanation
func testDerived() {
	f := gen.OpenUCDFile("DerivedNormalizationProps.txt")
	defer f.Close()
	p := ucd.New(f)
	for p.Next() {
		r := p.Rune(0)
		c := &chars[r]

		var ftype, mode int
		qt := p.String(1)
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
			continue
		}
		var qr QCResult
		switch p.String(2) {
		case "Y":
			qr = QCYes
		case "N":
			qr = QCNo
		case "M":
			qr = QCMaybe
		default:
			log.Fatalf(`Unexpected quick check value "%s"`, p.String(2))
		}
		if got := c.forms[ftype].quickCheck[mode]; got != qr {
			log.Printf("%U: FAILED %s (was %v need %v)\n", r, qt, got, qr)
		}
		c.forms[ftype].verified[mode] = true
	}
	if err := p.Err(); err != nil {
		log.Fatal(err)
	}
	// Any unspecified value must be QCYes. Verify this.
	for i, c := range chars {
		for j, fd := range c.forms {
			for k, qr := range fd.quickCheck {
				if !fd.verified[k] && qr != QCYes {
					m := "%U: FAIL F:%d M:%d (was %v need Yes) %s\n"
					log.Printf(m, i, j, k, qr, c.name)
				}
			}
		}
	}
}

var testHeader = `const (
	Yes = iota
	No
	Maybe
)

type formData struct {
	qc              uint8
	combinesForward bool
	decomposition   string
}

type runeData struct {
	r      rune
	ccc    uint8
	nLead  uint8
	nTrail uint8
	f      [2]formData // 0: canonical; 1: compatibility
}

func f(qc uint8, cf bool, dec string) [2]formData {
	return [2]formData{{qc, cf, dec}, {qc, cf, dec}}
}

func g(qc, qck uint8, cf, cfk bool, d, dk string) [2]formData {
	return [2]formData{{qc, cf, d}, {qck, cfk, dk}}
}

var testData = []runeData{
`

func printTestdata() {
	type lastInfo struct {
		ccc    uint8
		nLead  uint8
		nTrail uint8
		f      string
	}

	last := lastInfo{}
	w := &bytes.Buffer{}
	fmt.Fprintf(w, testHeader)
	for r, c := range chars {
		f := c.forms[FCanonical]
		qc, cf, d := f.quickCheck[MComposed], f.combinesForward, string(f.expandedDecomp)
		f = c.forms[FCompatibility]
		qck, cfk, dk := f.quickCheck[MComposed], f.combinesForward, string(f.expandedDecomp)
		s := ""
		if d == dk && qc == qck && cf == cfk {
			s = fmt.Sprintf("f(%s, %v, %q)", qc, cf, d)
		} else {
			s = fmt.Sprintf("g(%s, %s, %v, %v, %q, %q)", qc, qck, cf, cfk, d, dk)
		}
		current := lastInfo{c.ccc, c.nLeadingNonStarters, c.nTrailingNonStarters, s}
		if last != current {
			fmt.Fprintf(w, "\t{0x%x, %d, %d, %d, %s},\n", r, c.origCCC, c.nLeadingNonStarters, c.nTrailingNonStarters, s)
			last = current
		}
	}
	fmt.Fprintln(w, "}")
	gen.WriteGoFile("data_test.go", "norm", w.Bytes())
}
