// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

import (
	"archive/zip"
	"bufio"
	"bytes"
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
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// This regression test runs tests for the test files in CollationTest.zip
// (taken from http://www.unicode.org/Public/UCA/<unicode.Version>/).
//
// The test files have the following form:
// # header
// 0009 0021;	# ('\u0009') <CHARACTER TABULATION>	[| | | 0201 025E]
// 0009 003F;	# ('\u0009') <CHARACTER TABULATION>	[| | | 0201 0263]
// 000A 0021;	# ('\u000A') <LINE FEED (LF)>	[| | | 0202 025E]
// 000A 003F;	# ('\u000A') <LINE FEED (LF)>	[| | | 0202 0263]
//
// The part before the semicolon is the hex representation of a sequence
// of runes. After the hash mark is a comment. The strings
// represented by rune sequence are in the file in sorted order, as
// defined by the DUCET.

var testdata = flag.String("testdata",
	"http://www.unicode.org/Public/UCA/"+unicode.Version+"/CollationTest.zip",
	"URL of Unicode collation tests zip file")
var ducet = flag.String("ducet",
	"http://unicode.org/Public/UCA/"+unicode.Version+"/allkeys.txt",
	"URL of the Default Unicode Collation Element Table (DUCET).")
var localFiles = flag.Bool("local",
	false,
	"data files have been copied to the current directory; for debugging only")

type Test struct {
	name    string
	str     [][]byte
	comment []string
}

var versionRe = regexp.MustCompile(`# UCA Version: (.*)\n?$`)
var testRe = regexp.MustCompile(`^([\dA-F ]+);.*# (.*)\n?$`)

func Error(e error) {
	if e != nil {
		log.Fatal(e)
	}
}

// openReader opens the url or file given by url and returns it as an io.ReadCloser
// or nil on error.
func openReader(url string) io.ReadCloser {
	if *localFiles {
		pwd, _ := os.Getwd()
		url = "file://" + path.Join(pwd, path.Base(url))
	}
	t := &http.Transport{}
	t.RegisterProtocol("file", http.NewFileTransport(http.Dir("/")))
	c := &http.Client{Transport: t}
	resp, err := c.Get(url)
	Error(err)
	if resp.StatusCode != 200 {
		Error(fmt.Errorf(`bad GET status for "%s": %s`, url, resp.Status))
	}
	return resp.Body
}

// parseUCA parses a Default Unicode Collation Element Table of the format
// specified in http://www.unicode.org/reports/tr10/#File_Format.
// It returns the variable top.
func parseUCA(builder *build.Builder) {
	r := openReader(*ducet)
	defer r.Close()
	input := bufio.NewReader(r)
	colelem := regexp.MustCompile(`\[([.*])([0-9A-F.]+)\]`)
	for i := 1; true; i++ {
		l, prefix, err := input.ReadLine()
		if err == io.EOF {
			break
		}
		Error(err)
		line := string(l)
		if prefix {
			log.Fatalf("%d: buffer overflow", i)
		}
		if len(line) == 0 || line[0] == '#' {
			continue
		}
		if line[0] == '@' {
			if strings.HasPrefix(line[1:], "version ") {
				if v := strings.Split(line[1:], " ")[1]; v != unicode.Version {
					log.Fatalf("incompatible version %s; want %s", v, unicode.Version)
				}
			}
		} else {
			// parse entries
			part := strings.Split(line, " ; ")
			if len(part) != 2 {
				log.Fatalf("%d: production rule without ';': %v", i, line)
			}
			lhs := []rune{}
			for _, v := range strings.Split(part[0], " ") {
				if v != "" {
					lhs = append(lhs, rune(convHex(i, v)))
				}
			}
			vars := []int{}
			rhs := [][]int{}
			for i, m := range colelem.FindAllStringSubmatch(part[1], -1) {
				if m[1] == "*" {
					vars = append(vars, i)
				}
				elem := []int{}
				for _, h := range strings.Split(m[2], ".") {
					elem = append(elem, convHex(i, h))
				}
				rhs = append(rhs, elem)
			}
			builder.Add(lhs, rhs, vars)
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

func loadTestData() []Test {
	f := openReader(*testdata)
	buffer, err := ioutil.ReadAll(f)
	f.Close()
	Error(err)
	archive, err := zip.NewReader(bytes.NewReader(buffer), int64(len(buffer)))
	Error(err)
	tests := []Test{}
	for _, f := range archive.File {
		// Skip the short versions, which are simply duplicates of the long versions.
		if strings.Contains(f.Name, "SHORT") || f.FileInfo().IsDir() {
			continue
		}
		ff, err := f.Open()
		Error(err)
		defer ff.Close()
		input := bufio.NewReader(ff)
		test := Test{name: path.Base(f.Name)}
		for {
			line, err := input.ReadString('\n')
			if err != nil {
				if err == io.EOF {
					break
				}
				log.Fatal(err)
			}
			if len(line) <= 1 || line[0] == '#' {
				if m := versionRe.FindStringSubmatch(line); m != nil {
					if m[1] != unicode.Version {
						log.Printf("warning:%s: version is %s; want %s", f.Name, m[1], unicode.Version)
					}
				}
				continue
			}
			m := testRe.FindStringSubmatch(line)
			if m == nil || len(m) < 3 {
				log.Fatalf(`Failed to parse: "%s" result: %#v`, line, m)
			}
			str := []byte{}
			// In the regression test data (unpaired) surrogates are assigned a weight
			// corresponding to their code point value.  However, utf8.DecodeRune,
			// which is used to compute the implicit weight, assigns FFFD to surrogates.
			// We therefore skip tests with surrogates.  This skips about 35 entries
			// per test.
			valid := true
			for _, split := range strings.Split(m[1], " ") {
				r, err := strconv.ParseUint(split, 16, 64)
				Error(err)
				valid = valid && utf8.ValidRune(rune(r))
				str = append(str, string(rune(r))...)
			}
			if valid {
				test.str = append(test.str, str)
				test.comment = append(test.comment, m[2])
			}
		}
		tests = append(tests, test)
	}
	return tests
}

var errorCount int

func fail(t Test, pattern string, args ...interface{}) {
	format := fmt.Sprintf("error:%s:%s", t.name, pattern)
	log.Printf(format, args...)
	errorCount++
	if errorCount > 30 {
		log.Fatal("too many errors")
	}
}

func runes(b []byte) []rune {
	return []rune(string(b))
}

func doTest(t Test) {
	bld := build.NewBuilder()
	parseUCA(bld)
	c, err := bld.Build()
	Error(err)
	c.Strength = collate.Tertiary
	c.Alternate = collate.AltShifted
	b := &collate.Buffer{}
	if strings.Contains(t.name, "NON_IGNOR") {
		c.Alternate = collate.AltNonIgnorable
	}
	prev := t.str[0]
	for i := 1; i < len(t.str); i++ {
		b.Reset()
		s := t.str[i]
		ka := c.Key(b, prev)
		kb := c.Key(b, s)
		if r := bytes.Compare(ka, kb); r == 1 {
			fail(t, "%d: Key(%.4X) < Key(%.4X) (%X < %X) == %d; want -1 or 0", i, []rune(string(prev)), []rune(string(s)), ka, kb, r)
			prev = s
			continue
		}
		if r := c.Compare(prev, s); r == 1 {
			fail(t, "%d: Compare(%.4X, %.4X) == %d; want -1 or 0", i, runes(prev), runes(s), r)
		}
		if r := c.Compare(s, prev); r == -1 {
			fail(t, "%d: Compare(%.4X, %.4X) == %d; want 1 or 0", i, runes(s), runes(prev), r)
		}
		prev = s
	}
}

func main() {
	flag.Parse()
	for _, test := range loadTestData() {
		doTest(test)
	}
	if errorCount == 0 {
		fmt.Println("PASS")
	}
}
