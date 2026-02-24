// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A stand-alone Go module that generates ../schema.go using the
// upstream Wycheproof JSON schema documents.
//
// We maintain this in a separate Go module and vendor the resulting
// generated .go code to avoid the standard library taking a direct
// dependency on the c2sp/wycheproof or atombender/go-jsonschema modules.

package main

import (
	"io/fs"
	"log"
	"os"
	"path/filepath"

	"github.com/atombender/go-jsonschema/pkg/generator"
	"github.com/c2sp/wycheproof"
)

func main() {
	ouputName := "schema.go"
	cfg := generator.Config{
		DefaultPackageName: "wycheproof",
		DefaultOutputName:  ouputName,
		Tags:               []string{"json"},
		Warner: func(message string) {
			log.Printf("go-jsonschema: %s", message)
		},
	}
	gen, err := generator.New(cfg)
	if err != nil {
		log.Fatal(err)
	}

	// Without upstream modifications we can't use the embedded Wycheproof
	// schema FS directly w/ go-jsonschema and instead make files in a tempdir
	// on the native FS. See https://github.com/omissis/go-jsonschema/issues/495
	schemaDir, err := os.MkdirTemp("", "*-wycheproof")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(schemaDir)

	rawSchemas, _ := fs.ReadDir(wycheproof.Schemas, ".")
	for _, entry := range rawSchemas {
		entryName := entry.Name()
		schemaFile := filepath.Join(schemaDir, entryName)

		schemaData, err := fs.ReadFile(wycheproof.Schemas, entryName)
		if err != nil {
			log.Fatalf("reading %s: %v", entryName, err)
		}
		err = os.WriteFile(schemaFile, schemaData, 0644)
		if err != nil {
			log.Fatalf("writing %s: %v", schemaFile, err)
		}
	}

	// Note: it's important we process schemas in a second pass after writing
	// **all** of the schema file content to disk so that go-jsonschema can
	// resolve cross-file references.
	for _, entry := range rawSchemas {
		entryName := entry.Name()
		schemaFile := filepath.Join(schemaDir, entryName)

		err = gen.DoFile(schemaFile)
		if err != nil {
			log.Fatalf("processing %s: %v", schemaFile, err)
		}
	}

	sources, err := gen.Sources()
	if err != nil {
		log.Fatalf("error generating sources: %v\n", err)
	}
	if sourceCount := len(sources); sourceCount != 1 {
		log.Fatalf("expected to generate 1 source file, got %d\n", sourceCount)
	}
	content, ok := sources[ouputName]
	if !ok {
		log.Fatalf("missing generated %q output file source", ouputName)
	}
	outFile := filepath.Join("../", ouputName)
	if err := os.WriteFile(outFile, content, 0644); err != nil {
		log.Fatalf("error writing file %s: %v\n", outFile, err)
	}
}
