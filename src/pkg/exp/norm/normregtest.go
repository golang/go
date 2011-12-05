// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
	"exp/norm"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

func main() {
	flag.Parse()
	loadTestData()
	CharacterByCharacterTests()
	StandardTests()
	PerformanceTest()
	if errorCount == 0 {
		fmt.Println("PASS")
	}
}

const file = "NormalizationTest.txt"

var url = flag.String("url",
	"http://www.unicode.org/Public/6.0.0/ucd/"+file,
	"URL of Unicode database directory")
var localFiles = flag.Bool("local",
	false,
	"data files have been copied to the current directory; for debugging only")

var logger = log.New(os.Stderr, "", log.Lshortfile)

// This regression test runs the test set in NormalizationTest.txt
// (taken from http://www.unicode.org/Public/6.0.0/ucd/).
//
// NormalizationTest.txt has form:
// @Part0 # Specific cases
// #
// 1E0A;1E0A;0044 0307;1E0A;0044 0307; # (Ḋ; Ḋ; D◌̇; Ḋ; D◌̇; ) LATIN CAPITAL LETTER D WITH DOT ABOVE
// 1E0C;1E0C;0044 0323;1E0C;0044 0323; # (Ḍ; Ḍ; D◌̣; Ḍ; D◌̣; ) LATIN CAPITAL LETTER D WITH DOT BELOW
//
// Each test has 5 columns (c1, c2, c3, c4, c5), where 
// (c1, c2, c3, c4, c5) == (c1, NFC(c1), NFD(c1), NFKC(c1), NFKD(c1))
//
// CONFORMANCE:
// 1. The following invariants must be true for all conformant implementations
//
//    NFC
//      c2 ==  NFC(c1) ==  NFC(c2) ==  NFC(c3)
//      c4 ==  NFC(c4) ==  NFC(c5)
//
//    NFD
//      c3 ==  NFD(c1) ==  NFD(c2) ==  NFD(c3)
//      c5 ==  NFD(c4) ==  NFD(c5)
//
//    NFKC
//      c4 == NFKC(c1) == NFKC(c2) == NFKC(c3) == NFKC(c4) == NFKC(c5)
//
//    NFKD
//      c5 == NFKD(c1) == NFKD(c2) == NFKD(c3) == NFKD(c4) == NFKD(c5)
//
// 2. For every code point X assigned in this version of Unicode that is not
//    specifically listed in Part 1, the following invariants must be true
//    for all conformant implementations:
//
//      X == NFC(X) == NFD(X) == NFKC(X) == NFKD(X)
//

// Column types.
const (
	cRaw = iota
	cNFC
	cNFD
	cNFKC
	cNFKD
	cMaxColumns
)

// Holds data from NormalizationTest.txt
var part []Part

type Part struct {
	name   string
	number int
	tests  []Test
}

type Test struct {
	name   string
	partnr int
	number int
	r      rune                // used for character by character test
	cols   [cMaxColumns]string // Each has 5 entries, see below.
}

func (t Test) Name() string {
	if t.number < 0 {
		return part[t.partnr].name
	}
	return fmt.Sprintf("%s:%d", part[t.partnr].name, t.number)
}

var partRe = regexp.MustCompile(`@Part(\d) # (.*)\n$`)
var testRe = regexp.MustCompile(`^` + strings.Repeat(`([\dA-F ]+);`, 5) + ` # (.*)\n?$`)

var counter int

// Load the data form NormalizationTest.txt
func loadTestData() {
	if *localFiles {
		pwd, _ := os.Getwd()
		*url = "file://" + path.Join(pwd, file)
	}
	t := &http.Transport{}
	t.RegisterProtocol("file", http.NewFileTransport(http.Dir("/")))
	c := &http.Client{Transport: t}
	resp, err := c.Get(*url)
	if err != nil {
		logger.Fatal(err)
	}
	if resp.StatusCode != 200 {
		logger.Fatal("bad GET status for "+file, resp.Status)
	}
	f := resp.Body
	defer f.Close()
	input := bufio.NewReader(f)
	for {
		line, err := input.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			logger.Fatal(err)
		}
		if len(line) == 0 || line[0] == '#' {
			continue
		}
		m := partRe.FindStringSubmatch(line)
		if m != nil {
			if len(m) < 3 {
				logger.Fatal("Failed to parse Part: ", line)
			}
			i, err := strconv.Atoi(m[1])
			if err != nil {
				logger.Fatal(err)
			}
			name := m[2]
			part = append(part, Part{name: name[:len(name)-1], number: i})
			continue
		}
		m = testRe.FindStringSubmatch(line)
		if m == nil || len(m) < 7 {
			logger.Fatalf(`Failed to parse: "%s" result: %#v`, line, m)
		}
		test := Test{name: m[6], partnr: len(part) - 1, number: counter}
		counter++
		for j := 1; j < len(m)-1; j++ {
			for _, split := range strings.Split(m[j], " ") {
				r, err := strconv.ParseUint(split, 16, 64)
				if err != nil {
					logger.Fatal(err)
				}
				if test.r == 0 {
					// save for CharacterByCharacterTests
					test.r = int(r)
				}
				var buf [utf8.UTFMax]byte
				sz := utf8.EncodeRune(buf[:], rune(r))
				test.cols[j-1] += string(buf[:sz])
			}
		}
		part := &part[len(part)-1]
		part.tests = append(part.tests, test)
	}
}

var fstr = []string{"NFC", "NFD", "NFKC", "NFKD"}

var errorCount int

func cmpResult(t *Test, name string, f norm.Form, gold, test, result string) {
	if gold != result {
		errorCount++
		if errorCount > 20 {
			return
		}
		st, sr, sg := []rune(test), []rune(result), []rune(gold)
		logger.Printf("%s:%s: %s(%X)=%X; want:%X: %s",
			t.Name(), name, fstr[f], st, sr, sg, t.name)
	}
}

func cmpIsNormal(t *Test, name string, f norm.Form, test string, result, want bool) {
	if result != want {
		errorCount++
		if errorCount > 20 {
			return
		}
		logger.Printf("%s:%s: %s(%X)=%v; want: %v", t.Name(), name, fstr[f], []rune(test), result, want)
	}
}

func doTest(t *Test, f norm.Form, gold, test string) {
	result := f.Bytes([]byte(test))
	cmpResult(t, "Bytes", f, gold, test, string(result))
	for i := range test {
		out := f.Append(f.Bytes([]byte(test[:i])), []byte(test[i:])...)
		cmpResult(t, fmt.Sprintf(":Append:%d", i), f, gold, test, string(out))
	}
	cmpIsNormal(t, "IsNormal", f, test, f.IsNormal([]byte(test)), test == gold)
}

func doConformanceTests(t *Test, partn int) {
	for i := 0; i <= 2; i++ {
		doTest(t, norm.NFC, t.cols[1], t.cols[i])
		doTest(t, norm.NFD, t.cols[2], t.cols[i])
		doTest(t, norm.NFKC, t.cols[3], t.cols[i])
		doTest(t, norm.NFKD, t.cols[4], t.cols[i])
	}
	for i := 3; i <= 4; i++ {
		doTest(t, norm.NFC, t.cols[3], t.cols[i])
		doTest(t, norm.NFD, t.cols[4], t.cols[i])
		doTest(t, norm.NFKC, t.cols[3], t.cols[i])
		doTest(t, norm.NFKD, t.cols[4], t.cols[i])
	}
}

func CharacterByCharacterTests() {
	tests := part[1].tests
	last := 0
	for i := 0; i <= len(tests); i++ { // last one is special case
		var r int
		if i == len(tests) {
			r = 0x2FA1E // Don't have to go to 0x10FFFF
		} else {
			r = tests[i].r
		}
		for last++; last < r; last++ {
			// Check all characters that were not explicitly listed in the test.
			t := &Test{partnr: 1, number: -1}
			char := string(last)
			doTest(t, norm.NFC, char, char)
			doTest(t, norm.NFD, char, char)
			doTest(t, norm.NFKC, char, char)
			doTest(t, norm.NFKD, char, char)
		}
		if i < len(tests) {
			doConformanceTests(&tests[i], 1)
		}
	}
}

func StandardTests() {
	for _, j := range []int{0, 2, 3} {
		for _, test := range part[j].tests {
			doConformanceTests(&test, j)
		}
	}
}

// PerformanceTest verifies that normalization is O(n). If any of the
// code does not properly check for maxCombiningChars, normalization
// may exhibit O(n**2) behavior.
func PerformanceTest() {
	runtime.GOMAXPROCS(2)
	success := make(chan bool, 1)
	go func() {
		buf := bytes.Repeat([]byte("\u035D"), 1024*1024)
		buf = append(buf, "\u035B"...)
		norm.NFC.Append(nil, buf...)
		success <- true
	}()
	timeout := time.After(1e9)
	select {
	case <-success:
		// test completed before the timeout
	case <-timeout:
		errorCount++
		logger.Printf(`unexpectedly long time to complete PerformanceTest`)
	}
}
