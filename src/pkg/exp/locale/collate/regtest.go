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

var url = flag.String("url",
	"http://www.unicode.org/Public/UCA/"+unicode.Version+"/CollationTest.zip",
	"URL of Unicode collation tests zip file")
var localFiles = flag.Bool("local",
	false,
	"data files have been copied to the current directory; for debugging only")

type Test struct {
	name    string
	str     []string
	comment []string
}

var versionRe = regexp.MustCompile(`# UCA Version: (.*)\n?$`)
var testRe = regexp.MustCompile(`^([\dA-F ]+);.*# (.*)\n?$`)

func Error(e error) {
	if e != nil {
		log.Fatal(e)
	}
}

func loadTestData() []Test {
	if *localFiles {
		pwd, _ := os.Getwd()
		*url = "file://" + path.Join(pwd, path.Base(*url))
	}
	t := &http.Transport{}
	t.RegisterProtocol("file", http.NewFileTransport(http.Dir("/")))
	c := &http.Client{Transport: t}
	resp, err := c.Get(*url)
	Error(err)
	if resp.StatusCode != 200 {
		log.Fatalf(`bad GET status for "%s": %s`, *url, resp.Status)
	}
	f := resp.Body
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
			str := ""
			for _, split := range strings.Split(m[1], " ") {
				r, err := strconv.ParseUint(split, 16, 64)
				Error(err)
				str += string(rune(r))
			}
			test.str = append(test.str, str)
			test.comment = append(test.comment, m[2])
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
	c := collate.Root
	c.Strength = collate.Tertiary
	b := &collate.Buffer{}
	if strings.Contains(t.name, "NON_IGNOR") {
		c.Alternate = collate.AltNonIgnorable
	}

	prev := []byte(t.str[0])
	for i := 1; i < len(t.str); i++ {
		s := []byte(t.str[i])
		ka := c.Key(b, prev)
		kb := c.Key(b, s)
		if r := bytes.Compare(ka, kb); r == 1 {
			fail(t, "%d: Key(%.4X) < Key(%.4X) (%X < %X) == %d; want -1 or 0", i, runes(prev), runes(s), ka, kb, r)
			prev = s
			continue
		}
		if r := c.Compare(b, prev, s); r == 1 {
			fail(t, "%d: Compare(%.4X, %.4X) == %d; want -1 or 0", i, runes(prev), runes(s), r)
		}
		if r := c.Compare(b, s, prev); r == -1 {
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
