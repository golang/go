// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// Collation table generator.
// Data read from the web.

package main

import (
	"bufio"
	"bytes"
	"exp/locale/collate"
	"exp/locale/collate/build"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

var ducet = flag.String("ducet",
	"http://unicode.org/Public/UCA/"+unicode.Version+"/allkeys.txt",
	"URL of the Default Unicode Collation Element Table (DUCET).")
var test = flag.Bool("test",
	false,
	"test existing tables; can be used to compare web data with package data")
var localFiles = flag.Bool("local",
	false,
	"data files have been copied to the current directory; for debugging only")

func failOnError(e error) {
	if e != nil {
		log.Fatal(e)
	}
}

// openReader opens the url or file given by url and returns it as an io.ReadCloser
// or nil on error.
func openReader(url string) (io.ReadCloser, error) {
	if *localFiles {
		pwd, _ := os.Getwd()
		url = "file://" + path.Join(pwd, path.Base(url))
	}
	t := &http.Transport{}
	t.RegisterProtocol("file", http.NewFileTransport(http.Dir("/")))
	c := &http.Client{Transport: t}
	resp, err := c.Get(url)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf(`bad GET status for "%s": %s`, url, resp.Status)
	}
	return resp.Body, nil
}

// parseUCA parses a Default Unicode Collation Element Table of the format
// specified in http://www.unicode.org/reports/tr10/#File_Format.
// It returns the variable top.
func parseUCA(builder *build.Builder) {
	r, err := openReader(*ducet)
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

type stringSet struct {
	set []string
}

func (ss *stringSet) add(s string) {
	ss.set = append(ss.set, s)
}

func (ss *stringSet) values() []string {
	ss.compact()
	return ss.set
}

func (ss *stringSet) compact() {
	a := ss.set
	sort.Strings(a)
	k := 0
	for i := 1; i < len(a); i++ {
		if a[k] != a[i] {
			a[k+1] = a[i]
			k++
		}
	}
	ss.set = a[:k+1]
}

func testCollator(c *collate.Collator) {
	c0 := collate.Root

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
		if bytes.Compare(k0, k) != 0 {
			failOnError(fmt.Errorf("test:%U: keys differ (%x vs %x)", []rune(str), k0, k))
		}
		buf.ResetKeys()
	}
	fmt.Println("PASS")
}

// TODO: move this functionality to exp/locale/collate/build.
func printCollators(c *collate.Collator) {
	const name = "Root"
	fmt.Printf("var _%s = Collator{\n", name)
	fmt.Printf("\tStrength: %v,\n", c.Strength)
	fmt.Printf("\tf: norm.NFD,\n")
	fmt.Printf("\tt: &%sTable,\n", strings.ToLower(name))
	fmt.Printf("}\n\n")
	fmt.Printf("var (\n")
	fmt.Printf("\t%s = _%s\n", name, name)
	fmt.Printf(")\n\n")
}

func main() {
	flag.Parse()
	b := build.NewBuilder()
	parseUCA(b)
	c, err := b.Build()
	failOnError(err)

	if *test {
		testCollator(c)
	} else {
		fmt.Println("// Generated by running")
		fmt.Printf("//  maketables --ducet=%s\n", *ducet)
		fmt.Println("// DO NOT EDIT")
		fmt.Println("// TODO: implement more compact representation for sparse blocks.")
		fmt.Println("")
		fmt.Println("package collate")
		fmt.Println("")
		fmt.Println(`import "exp/norm"`)
		fmt.Println("")

		printCollators(c)

		_, err = b.Print(os.Stdout)
		failOnError(err)
	}
}
