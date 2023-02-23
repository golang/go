// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.19
// +build go1.19

package main

import (
	"bytes"
	"fmt"
	"log"
	"strings"
)

// a newType is a type that needs a name and a definition
// These are the various types that the json specification doesn't name
type newType struct {
	name       string
	properties Properties // for struct/literal types
	items      []*Type    // for other types ("and", "tuple")
	line       int
	kind       string // Or, And, Tuple, Lit, Map
	typ        *Type
}

func generateDoc(out *bytes.Buffer, doc string) {
	if doc == "" {
		return
	}

	if !strings.Contains(doc, "\n") {
		fmt.Fprintf(out, "// %s\n", doc)
		return
	}
	var list bool
	for _, line := range strings.Split(doc, "\n") {
		// Lists in metaModel.json start with a dash.
		// To make a go doc list they have to be preceded
		// by a blank line, and indented.
		// (see type TextDccumentFilter in protocol.go)
		if len(line) > 0 && line[0] == '-' {
			if !list {
				list = true
				fmt.Fprintf(out, "//\n")
			}
			fmt.Fprintf(out, "//  %s\n", line)
		} else {
			if len(line) == 0 {
				list = false
			}
			fmt.Fprintf(out, "// %s\n", line)
		}
	}
}

// decide if a property is optional, and if it needs a *
// return ",omitempty" if it is optional, and "*" if it needs a pointer
func propStar(name string, t NameType, gotype string) (string, string) {
	var opt, star string
	if t.Optional {
		star = "*"
		opt = ",omitempty"
	}
	if strings.HasPrefix(gotype, "[]") || strings.HasPrefix(gotype, "map[") {
		star = "" // passed by reference, so no need for *
	} else {
		switch gotype {
		case "bool", "uint32", "int32", "string", "interface{}":
			star = "" // gopls compatibility if t.Optional
		}
	}
	ostar, oopt := star, opt
	if newStar, ok := goplsStar[prop{name, t.Name}]; ok {
		switch newStar {
		case nothing:
			star, opt = "", ""
		case wantStar:
			star, opt = "*", ""
		case wantOpt:
			star, opt = "", ",omitempty"
		case wantOptStar:
			star, opt = "*", ",omitempty"
		}
		if star == ostar && opt == oopt { // no change
			log.Printf("goplsStar[ {%q, %q} ](%d) useless %s/%s %s/%s", name, t.Name, t.Line, ostar, star, oopt, opt)
		}
		usedGoplsStar[prop{name, t.Name}] = true
	}

	return opt, star
}

func goName(s string) string {
	// Go naming conventions
	if strings.HasSuffix(s, "Id") {
		s = s[:len(s)-len("Id")] + "ID"
	} else if strings.HasSuffix(s, "Uri") {
		s = s[:len(s)-3] + "URI"
	} else if s == "uri" {
		s = "URI"
	} else if s == "id" {
		s = "ID"
	}

	// renames for temporary GOPLS compatibility
	if news := goplsType[s]; news != "" {
		usedGoplsType[s] = true
		s = news
	}
	// Names beginning _ are not exported
	if strings.HasPrefix(s, "_") {
		s = strings.Replace(s, "_", "X", 1)
	}
	if s != "string" { // base types are unchanged (textDocuemnt/diagnostic)
		// Title is deprecated, but a) s is only one word, b) replacement is too heavy-weight
		s = strings.Title(s)
	}
	return s
}
