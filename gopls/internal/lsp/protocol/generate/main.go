// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.19
// +build go1.19

package main

import (
	"flag"
	"fmt"
	"log"
	"os"
)

var (
	// git clone https://github.com/microsoft/vscode-languageserver-node.git
	repodir   = flag.String("d", "", "directory of vscode-languageserver-node")
	outputdir = flag.String("o", "gen", "output directory")
	cmpolder  = flag.String("c", "", "directory of older generated code")
)

func main() {
	log.SetFlags(log.Lshortfile) // log file name and line number, not time
	flag.Parse()

	if *repodir == "" {
		*repodir = fmt.Sprintf("%s/vscode-languageserver-node", os.Getenv("HOME"))
	}
	spec := parse(*repodir)

	// index the information in the specification
	spec.indexRPCInfo() // messages
	spec.indexDefInfo() // named types

}

func (s *spec) indexRPCInfo() {
	for _, r := range s.model.Requests {
		r := r
		s.byMethod[r.Method] = &r
	}
	for _, n := range s.model.Notifications {
		n := n
		if n.Method == "$/cancelRequest" {
			// viewed as too confusing to generate
			continue
		}
		s.byMethod[n.Method] = &n
	}
}

func (sp *spec) indexDefInfo() {
	for _, s := range sp.model.Structures {
		s := s
		sp.byName[s.Name] = &s
	}
	for _, e := range sp.model.Enumerations {
		e := e
		sp.byName[e.Name] = &e
	}
	for _, ta := range sp.model.TypeAliases {
		ta := ta
		sp.byName[ta.Name] = &ta
	}

	// some Structure and TypeAlias names need to be changed for Go
	// so byName contains the name used in the .json file, and
	// the Name field contains the Go version of the name.
	v := sp.model.Structures
	for i, s := range v {
		switch s.Name {
		case "_InitializeParams": // _ is not upper case
			v[i].Name = "XInitializeParams"
		case "ConfigurationParams": // gopls compatibility
			v[i].Name = "ParamConfiguration"
		case "InitializeParams": // gopls compatibility
			v[i].Name = "ParamInitialize"
		case "PreviousResultId": // Go naming convention
			v[i].Name = "PreviousResultID"
		case "WorkspaceFoldersServerCapabilities": // gopls compatibility
			v[i].Name = "WorkspaceFolders5Gn"
		}
	}
	w := sp.model.TypeAliases
	for i, t := range w {
		switch t.Name {
		case "PrepareRenameResult": // gopls compatibility
			w[i].Name = "PrepareRename2Gn"
		}
	}
}
