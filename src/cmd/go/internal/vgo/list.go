// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vgo

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"regexp"
	"unicode/utf8"

	"cmd/go/internal/base"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/module"
)

func ListT(pkgs []string) {
	if Init(); !Enabled() {
		base.Fatalf("go list: cannot use -t outside module")
	}
	InitMod()

	if len(pkgs) == 0 {
		base.Fatalf("vgo list -t: need list of modules")
	}

	for _, pkg := range pkgs {
		repo, err := modfetch.Lookup(pkg)
		if err != nil {
			base.Errorf("vgo list -t: %v", err)
			continue
		}
		path := repo.ModulePath()
		fmt.Printf("%s\n", path)
		tags, err := repo.Versions("")
		if err != nil {
			base.Errorf("vgo list -t: %v", err)
			continue
		}
		for _, t := range tags {
			if excluded[module.Version{Path: path, Version: t}] {
				t += " # excluded"
			}
			fmt.Printf("\t%s\n", t)
		}
	}
}

func ListM() {
	if Init(); !Enabled() {
		base.Fatalf("go list: cannot use -m outside module")
	}
	InitMod()
	iterate(func(*loader) {})
	printListM(os.Stdout)
}

func printListM(w io.Writer) {
	var rows [][]string
	rows = append(rows, []string{"MODULE", "VERSION"})
	for _, mod := range buildList {
		v := mod.Version
		if v == "" {
			v = "-"
		}
		rows = append(rows, []string{mod.Path, v})
		if r := replaced(mod); r != nil {
			rows = append(rows, []string{" => " + r.New.Path, r.New.Version})
		}
	}
	printTable(w, rows)
}

func ListMU() {
	if Init(); !Enabled() {
		base.Fatalf("go list: cannot use -m outside module")
	}
	InitMod()

	quietLookup = true // do not chatter in v.Lookup
	iterate(func(*loader) {})

	var rows [][]string
	rows = append(rows, []string{"MODULE", "VERSION", "LATEST"})
	for _, mod := range buildList {
		var latest string
		v := mod.Version
		if v == "" {
			v = "-"
			latest = "-"
		} else {
			info, err := modfetch.Query(mod.Path, "latest", allowed)
			if err != nil {
				latest = "ERR: " + err.Error()
			} else {
				latest = info.Version
				if !isPseudoVersion(latest) && !info.Time.IsZero() {
					latest += info.Time.Local().Format(" (2006-01-02 15:04)")
				}
			}
			if !isPseudoVersion(mod.Version) {
				if info, err := modfetch.Query(mod.Path, mod.Version, nil); err == nil && !info.Time.IsZero() {
					v += info.Time.Local().Format(" (2006-01-02 15:04)")
				}
			}
		}
		if latest == v {
			latest = "-"
		}
		rows = append(rows, []string{mod.Path, v, latest})
	}
	printTable(os.Stdout, rows)
}

var pseudoVersionRE = regexp.MustCompile(`^v[0-9]+\.0\.0-[0-9]{14}-[A-Za-z0-9]+$`)

func isPseudoVersion(v string) bool {
	return pseudoVersionRE.MatchString(v)
}

func printTable(w io.Writer, rows [][]string) {
	var max []int
	for _, row := range rows {
		for i, c := range row {
			n := utf8.RuneCountInString(c)
			if i >= len(max) {
				max = append(max, n)
			} else if max[i] < n {
				max[i] = n
			}
		}
	}

	b := bufio.NewWriter(w)
	for _, row := range rows {
		for len(row) > 0 && row[len(row)-1] == "" {
			row = row[:len(row)-1]
		}
		for i, c := range row {
			b.WriteString(c)
			if i+1 < len(row) {
				for j := utf8.RuneCountInString(c); j < max[i]+2; j++ {
					b.WriteRune(' ')
				}
			}
		}
		b.WriteRune('\n')
	}
	b.Flush()
}
