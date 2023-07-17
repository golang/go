// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.19
// +build go1.19

// The generate command generates Go declarations from VSCode's
// description of the Language Server Protocol.
//
// To run it, type 'go generate' in the parent (protocol) directory.
package main

// see https://github.com/golang/go/issues/61217 for discussion of an issue

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/format"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const vscodeRepo = "https://github.com/microsoft/vscode-languageserver-node"

// lspGitRef names a branch or tag in vscodeRepo.
// It implicitly determines the protocol version of the LSP used by gopls.
// For example, tag release/protocol/3.17.3 of the repo defines protocol version 3.17.0.
// (Point releases are reflected in the git tag version even when they are cosmetic
// and don't change the protocol.)
var lspGitRef = "release/protocol/3.17.4-next.2"

var (
	repodir   = flag.String("d", "", "directory containing clone of "+vscodeRepo)
	outputdir = flag.String("o", ".", "output directory")
	// PJW: not for real code
	cmpdir      = flag.String("c", "", "directory of earlier code")
	doboth      = flag.String("b", "", "generate and compare")
	lineNumbers = flag.Bool("l", false, "add line numbers to generated output")
)

func main() {
	log.SetFlags(log.Lshortfile) // log file name and line number, not time
	flag.Parse()

	processinline()
}

func processinline() {
	// A local repository may be specified during debugging.
	// The default behavior is to download the canonical version.
	if *repodir == "" {
		tmpdir, err := os.MkdirTemp("", "")
		if err != nil {
			log.Fatal(err)
		}
		defer os.RemoveAll(tmpdir) // ignore error

		// Clone the repository.
		cmd := exec.Command("git", "clone", "--quiet", "--depth=1", "-c", "advice.detachedHead=false", vscodeRepo, "--branch="+lspGitRef, "--single-branch", tmpdir)
		cmd.Stdout = os.Stderr
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			log.Fatal(err)
		}

		*repodir = tmpdir
	} else {
		lspGitRef = fmt.Sprintf("(not git, local dir %s)", *repodir)
	}

	model := parse(filepath.Join(*repodir, "protocol/metaModel.json"))

	findTypeNames(model)
	generateOutput(model)

	fileHdr = fileHeader(model)

	// write the files
	writeclient()
	writeserver()
	writeprotocol()
	writejsons()

	checkTables()
}

// common file header for output files
var fileHdr string

func writeclient() {
	out := new(bytes.Buffer)
	fmt.Fprintln(out, fileHdr)
	out.WriteString(
		`import (
	"context"
	"encoding/json"

	"golang.org/x/tools/internal/jsonrpc2"
)
`)
	out.WriteString("type Client interface {\n")
	for _, k := range cdecls.keys() {
		out.WriteString(cdecls[k])
	}
	out.WriteString("}\n\n")
	out.WriteString("func clientDispatch(ctx context.Context, client Client, reply jsonrpc2.Replier, r jsonrpc2.Request) (bool, error) {\n")
	out.WriteString("\tswitch r.Method() {\n")
	for _, k := range ccases.keys() {
		out.WriteString(ccases[k])
	}
	out.WriteString(("\tdefault:\n\t\treturn false, nil\n\t}\n}\n\n"))
	for _, k := range cfuncs.keys() {
		out.WriteString(cfuncs[k])
	}

	x, err := format.Source(out.Bytes())
	if err != nil {
		os.WriteFile("/tmp/a.go", out.Bytes(), 0644)
		log.Fatalf("tsclient.go: %v", err)
	}

	if err := os.WriteFile(filepath.Join(*outputdir, "tsclient.go"), x, 0644); err != nil {
		log.Fatalf("%v writing tsclient.go", err)
	}
}

func writeserver() {
	out := new(bytes.Buffer)
	fmt.Fprintln(out, fileHdr)
	out.WriteString(
		`import (
	"context"
	"encoding/json"

	"golang.org/x/tools/internal/jsonrpc2"
)
`)
	out.WriteString("type Server interface {\n")
	for _, k := range sdecls.keys() {
		out.WriteString(sdecls[k])
	}
	out.WriteString(`	NonstandardRequest(ctx context.Context, method string, params interface{}) (interface{}, error)
}

func serverDispatch(ctx context.Context, server Server, reply jsonrpc2.Replier, r jsonrpc2.Request) (bool, error) {
	switch r.Method() {
`)
	for _, k := range scases.keys() {
		out.WriteString(scases[k])
	}
	out.WriteString(("\tdefault:\n\t\treturn false, nil\n\t}\n}\n\n"))
	for _, k := range sfuncs.keys() {
		out.WriteString(sfuncs[k])
	}
	out.WriteString(`func (s *serverDispatcher) NonstandardRequest(ctx context.Context, method string, params interface{}) (interface{}, error) {
	var result interface{}
	if err := s.sender.Call(ctx, method, params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
`)

	x, err := format.Source(out.Bytes())
	if err != nil {
		os.WriteFile("/tmp/a.go", out.Bytes(), 0644)
		log.Fatalf("tsserver.go: %v", err)
	}

	if err := os.WriteFile(filepath.Join(*outputdir, "tsserver.go"), x, 0644); err != nil {
		log.Fatalf("%v writing tsserver.go", err)
	}
}

func writeprotocol() {
	out := new(bytes.Buffer)
	fmt.Fprintln(out, fileHdr)
	out.WriteString("import \"encoding/json\"\n\n")

	// The followiing are unneeded, but make the new code a superset of the old
	hack := func(newer, existing string) {
		if _, ok := types[existing]; !ok {
			log.Fatalf("types[%q] not found", existing)
		}
		types[newer] = strings.Replace(types[existing], existing, newer, 1)
	}
	hack("ConfigurationParams", "ParamConfiguration")
	hack("InitializeParams", "ParamInitialize")
	hack("PreviousResultId", "PreviousResultID")
	hack("WorkspaceFoldersServerCapabilities", "WorkspaceFolders5Gn")
	hack("_InitializeParams", "XInitializeParams")
	// and some aliases to make the new code contain the old
	types["PrepareRename2Gn"] = "type PrepareRename2Gn = Msg_PrepareRename2Gn // (alias) line 13927\n"
	types["PrepareRenameResult"] = "type PrepareRenameResult = Msg_PrepareRename2Gn // (alias) line 13927\n"
	for _, k := range types.keys() {
		if k == "WatchKind" {
			types[k] = "type WatchKind = uint32 // line 13505" // strict gopls compatibility needs the '='
		}
		out.WriteString(types[k])
	}

	out.WriteString("\nconst (\n")
	for _, k := range consts.keys() {
		out.WriteString(consts[k])
	}
	out.WriteString(")\n\n")
	x, err := format.Source(out.Bytes())
	if err != nil {
		os.WriteFile("/tmp/a.go", out.Bytes(), 0644)
		log.Fatalf("tsprotocol.go: %v", err)
	}
	if err := os.WriteFile(filepath.Join(*outputdir, "tsprotocol.go"), x, 0644); err != nil {
		log.Fatalf("%v writing tsprotocol.go", err)
	}
}

func writejsons() {
	out := new(bytes.Buffer)
	fmt.Fprintln(out, fileHdr)
	out.WriteString("import \"encoding/json\"\n\n")
	out.WriteString("import \"fmt\"\n")

	out.WriteString(`
// UnmarshalError indicates that a JSON value did not conform to
// one of the expected cases of an LSP union type.
type UnmarshalError struct {
	msg string
}

func (e UnmarshalError) Error() string {
	return e.msg
}
`)

	for _, k := range jsons.keys() {
		out.WriteString(jsons[k])
	}
	x, err := format.Source(out.Bytes())
	if err != nil {
		os.WriteFile("/tmp/a.go", out.Bytes(), 0644)
		log.Fatalf("tsjson.go: %v", err)
	}
	if err := os.WriteFile(filepath.Join(*outputdir, "tsjson.go"), x, 0644); err != nil {
		log.Fatalf("%v writing tsjson.go", err)
	}
}

// create the common file header for the output files
func fileHeader(model Model) string {
	fname := filepath.Join(*repodir, ".git", "HEAD")
	buf, err := os.ReadFile(fname)
	if err != nil {
		log.Fatal(err)
	}
	buf = bytes.TrimSpace(buf)
	var githash string
	if len(buf) == 40 {
		githash = string(buf[:40])
	} else if bytes.HasPrefix(buf, []byte("ref: ")) {
		fname = filepath.Join(*repodir, ".git", string(buf[5:]))
		buf, err = os.ReadFile(fname)
		if err != nil {
			log.Fatal(err)
		}
		githash = string(buf[:40])
	} else {
		log.Fatalf("githash cannot be recovered from %s", fname)
	}

	format := `// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Code generated for LSP. DO NOT EDIT.

package protocol

// Code generated from %[1]s at ref %[2]s (hash %[3]s).
// %[4]s/blob/%[2]s/%[1]s
// LSP metaData.version = %[5]s.

`
	return fmt.Sprintf(format,
		"protocol/metaModel.json", // 1
		lspGitRef,                 // 2
		githash,                   // 3
		vscodeRepo,                // 4
		model.Version.Version)     // 5
}

func parse(fname string) Model {
	buf, err := os.ReadFile(fname)
	if err != nil {
		log.Fatal(err)
	}
	buf = addLineNumbers(buf)
	var model Model
	if err := json.Unmarshal(buf, &model); err != nil {
		log.Fatal(err)
	}
	return model
}

// Type.Value has to be treated specially for literals and maps
func (t *Type) UnmarshalJSON(data []byte) error {
	// First unmarshal only the unambiguous fields.
	var x struct {
		Kind    string  `json:"kind"`
		Items   []*Type `json:"items"`
		Element *Type   `json:"element"`
		Name    string  `json:"name"`
		Key     *Type   `json:"key"`
		Value   any     `json:"value"`
		Line    int     `json:"line"`
	}
	if err := json.Unmarshal(data, &x); err != nil {
		return err
	}
	*t = Type{
		Kind:    x.Kind,
		Items:   x.Items,
		Element: x.Element,
		Name:    x.Name,
		Value:   x.Value,
		Line:    x.Line,
	}

	// Then unmarshal the 'value' field based on the kind.
	// This depends on Unmarshal ignoring fields it doesn't know about.
	switch x.Kind {
	case "map":
		var x struct {
			Key   *Type `json:"key"`
			Value *Type `json:"value"`
		}
		if err := json.Unmarshal(data, &x); err != nil {
			return fmt.Errorf("Type.kind=map: %v", err)
		}
		t.Key = x.Key
		t.Value = x.Value

	case "literal":
		var z struct {
			Value ParseLiteral `json:"value"`
		}

		if err := json.Unmarshal(data, &z); err != nil {
			return fmt.Errorf("Type.kind=literal: %v", err)
		}
		t.Value = z.Value

	case "base", "reference", "array", "and", "or", "tuple",
		"stringLiteral":
		// no-op. never seen integerLiteral or booleanLiteral.

	default:
		return fmt.Errorf("cannot decode Type.kind %q: %s", x.Kind, data)
	}
	return nil
}

// which table entries were not used
func checkTables() {
	for k := range disambiguate {
		if !usedDisambiguate[k] {
			log.Printf("disambiguate[%v] unused", k)
		}
	}
	for k := range renameProp {
		if !usedRenameProp[k] {
			log.Printf("renameProp {%q, %q} unused", k[0], k[1])
		}
	}
	for k := range goplsStar {
		if !usedGoplsStar[k] {
			log.Printf("goplsStar {%q, %q} unused", k[0], k[1])
		}
	}
	for k := range goplsType {
		if !usedGoplsType[k] {
			log.Printf("unused goplsType[%q]->%s", k, goplsType[k])
		}
	}
}
