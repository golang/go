// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

// Package vulntest provides helpers for vulncheck functionality testing.
package vulntest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/txtar"
	"golang.org/x/vuln/client"
	"golang.org/x/vuln/osv"
)

// NewDatabase returns a read-only DB containing the provided
// txtar-format collection of vulnerability reports.
// Each vulnerability report is a YAML file whose format
// is defined in golang.org/x/vulndb/doc/format.md.
// A report file name must have the id as its base name,
// and have .yaml as its extension.
//
//	db, err := NewDatabase(ctx, reports)
//	...
//	defer db.Clean()
//	client, err := NewClient(db)
//	...
//
// The returned DB's Clean method must be called to clean up the
// generated database.
func NewDatabase(ctx context.Context, txtarReports []byte) (*DB, error) {
	disk, err := ioutil.TempDir("", "vulndb-test")
	if err != nil {
		return nil, err
	}
	if err := generateDB(ctx, txtarReports, disk, false); err != nil {
		os.RemoveAll(disk)
		return nil, err
	}

	return &DB{disk: disk}, nil
}

// DB is a read-only vulnerability database on disk.
// Users can use this database with golang.org/x/vuln APIs
// by setting the `VULNDB“ environment variable.
type DB struct {
	disk string
}

// URI returns the file URI that can be used for VULNDB environment
// variable.
func (db *DB) URI() string {
	u := span.URIFromPath(db.disk)
	return string(u)
}

// Clean deletes the database.
func (db *DB) Clean() error {
	return os.RemoveAll(db.disk)
}

// NewClient returns a vuln DB client that works with the given DB.
func NewClient(db *DB) (client.Client, error) {
	return client.NewClient([]string{db.URI()}, client.Options{})
}

//
// The following was selectively copied from golang.org/x/vulndb/internal/database
//

const (
	dbURL = "https://pkg.go.dev/vuln/"

	// idDirectory is the name of the directory that contains entries
	// listed by their IDs.
	idDirectory = "ID"

	// stdFileName is the name of the .json file in the vulndb repo
	// that will contain info on standard library vulnerabilities.
	stdFileName = "stdlib"

	// toolchainFileName is the name of the .json file in the vulndb repo
	// that will contain info on toolchain (cmd/...) vulnerabilities.
	toolchainFileName = "toolchain"

	// cmdModule is the name of the module containing Go toolchain
	// binaries.
	cmdModule = "cmd"

	// stdModule is the name of the module containing Go std packages.
	stdModule = "std"
)

// generateDB generates the file-based vuln DB in the directory jsonDir.
func generateDB(ctx context.Context, txtarData []byte, jsonDir string, indent bool) error {
	archive := txtar.Parse(txtarData)

	jsonVulns, entries, err := generateEntries(ctx, archive)
	if err != nil {
		return err
	}

	index := make(client.DBIndex, len(jsonVulns))
	for modulePath, vulns := range jsonVulns {
		epath, err := client.EscapeModulePath(modulePath)
		if err != nil {
			return err
		}
		if err := writeVulns(filepath.Join(jsonDir, epath), vulns, indent); err != nil {
			return err
		}
		for _, v := range vulns {
			if v.Modified.After(index[modulePath]) {
				index[modulePath] = v.Modified
			}
		}
	}
	if err := writeJSON(filepath.Join(jsonDir, "index.json"), index, indent); err != nil {
		return err
	}
	if err := writeAliasIndex(jsonDir, entries, indent); err != nil {
		return err
	}
	return writeEntriesByID(filepath.Join(jsonDir, idDirectory), entries, indent)
}

func generateEntries(_ context.Context, archive *txtar.Archive) (map[string][]osv.Entry, []osv.Entry, error) {
	now := time.Now()
	jsonVulns := map[string][]osv.Entry{}
	var entries []osv.Entry
	for _, f := range archive.Files {
		if !strings.HasSuffix(f.Name, ".yaml") {
			continue
		}
		r, err := readReport(bytes.NewReader(f.Data))
		if err != nil {
			return nil, nil, err
		}
		name := strings.TrimSuffix(filepath.Base(f.Name), filepath.Ext(f.Name))
		linkName := fmt.Sprintf("%s%s", dbURL, name)
		entry, modulePaths := generateOSVEntry(name, linkName, now, *r)
		for _, modulePath := range modulePaths {
			jsonVulns[modulePath] = append(jsonVulns[modulePath], entry)
		}
		entries = append(entries, entry)
	}
	return jsonVulns, entries, nil
}

func writeVulns(outPath string, vulns []osv.Entry, indent bool) error {
	if err := os.MkdirAll(filepath.Dir(outPath), 0755); err != nil {
		return fmt.Errorf("failed to create directory %q: %s", filepath.Dir(outPath), err)
	}
	return writeJSON(outPath+".json", vulns, indent)
}

func writeEntriesByID(idDir string, entries []osv.Entry, indent bool) error {
	// Write a directory containing entries by ID.
	if err := os.MkdirAll(idDir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %q: %v", idDir, err)
	}
	var idIndex []string
	for _, e := range entries {
		outPath := filepath.Join(idDir, e.ID+".json")
		if err := writeJSON(outPath, e, indent); err != nil {
			return err
		}
		idIndex = append(idIndex, e.ID)
	}
	// Write an index.json in the ID directory with a list of all the IDs.
	return writeJSON(filepath.Join(idDir, "index.json"), idIndex, indent)
}

// Write a JSON file containing a map from alias to GO IDs.
func writeAliasIndex(dir string, entries []osv.Entry, indent bool) error {
	aliasToGoIDs := map[string][]string{}
	for _, e := range entries {
		for _, a := range e.Aliases {
			aliasToGoIDs[a] = append(aliasToGoIDs[a], e.ID)
		}
	}
	return writeJSON(filepath.Join(dir, "aliases.json"), aliasToGoIDs, indent)
}

func writeJSON(filename string, value any, indent bool) (err error) {
	j, err := jsonMarshal(value, indent)
	if err != nil {
		return err
	}
	return os.WriteFile(filename, j, 0644)
}

func jsonMarshal(v any, indent bool) ([]byte, error) {
	if indent {
		return json.MarshalIndent(v, "", "  ")
	}
	return json.Marshal(v)
}

// generateOSVEntry create an osv.Entry for a report. In addition to the report, it
// takes the ID for the vuln and a URL that will point to the entry in the vuln DB.
// It returns the osv.Entry and a list of module paths that the vuln affects.
func generateOSVEntry(id, url string, lastModified time.Time, r Report) (osv.Entry, []string) {
	entry := osv.Entry{
		ID:        id,
		Published: r.Published,
		Modified:  lastModified,
		Withdrawn: r.Withdrawn,
		Details:   r.Description,
	}

	moduleMap := make(map[string]bool)
	for _, m := range r.Modules {
		switch m.Module {
		case stdModule:
			moduleMap[stdFileName] = true
		case cmdModule:
			moduleMap[toolchainFileName] = true
		default:
			moduleMap[m.Module] = true
		}
		entry.Affected = append(entry.Affected, generateAffected(m, url))
	}
	for _, ref := range r.References {
		entry.References = append(entry.References, osv.Reference{
			Type: string(ref.Type),
			URL:  ref.URL,
		})
	}

	var modulePaths []string
	for module := range moduleMap {
		modulePaths = append(modulePaths, module)
	}
	// TODO: handle missing fields - Aliases

	return entry, modulePaths
}

func generateAffectedRanges(versions []VersionRange) osv.Affects {
	a := osv.AffectsRange{Type: osv.TypeSemver}
	if len(versions) == 0 || versions[0].Introduced == "" {
		a.Events = append(a.Events, osv.RangeEvent{Introduced: "0"})
	}
	for _, v := range versions {
		if v.Introduced != "" {
			a.Events = append(a.Events, osv.RangeEvent{Introduced: v.Introduced.Canonical()})
		}
		if v.Fixed != "" {
			a.Events = append(a.Events, osv.RangeEvent{Fixed: v.Fixed.Canonical()})
		}
	}
	return osv.Affects{a}
}

func generateImports(m *Module) (imps []osv.EcosystemSpecificImport) {
	for _, p := range m.Packages {
		syms := append([]string{}, p.Symbols...)
		syms = append(syms, p.DerivedSymbols...)
		sort.Strings(syms)
		imps = append(imps, osv.EcosystemSpecificImport{
			Path:    p.Package,
			GOOS:    p.GOOS,
			GOARCH:  p.GOARCH,
			Symbols: syms,
		})
	}
	return imps
}
func generateAffected(m *Module, url string) osv.Affected {
	name := m.Module
	switch name {
	case stdModule:
		name = "stdlib"
	case cmdModule:
		name = "toolchain"
	}
	return osv.Affected{
		Package: osv.Package{
			Name:      name,
			Ecosystem: osv.GoEcosystem,
		},
		Ranges:           generateAffectedRanges(m.Versions),
		DatabaseSpecific: osv.DatabaseSpecific{URL: url},
		EcosystemSpecific: osv.EcosystemSpecific{
			Imports: generateImports(m),
		},
	}
}
