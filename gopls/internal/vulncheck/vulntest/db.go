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
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/gopls/internal/vulncheck/osv"
	"golang.org/x/tools/txtar"
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
	disk, err := os.MkdirTemp("", "vulndb-test")
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
	u := span.URIFromPath(filepath.Join(db.disk, "ID"))
	return string(u)
}

// Clean deletes the database.
func (db *DB) Clean() error {
	return os.RemoveAll(db.disk)
}

//
// The following was selectively copied from golang.org/x/vulndb/internal/database
//

const (
	dbURL = "https://pkg.go.dev/vuln/"

	// idDirectory is the name of the directory that contains entries
	// listed by their IDs.
	idDirectory = "ID"

	// cmdModule is the name of the module containing Go toolchain
	// binaries.
	cmdModule = "cmd"

	// stdModule is the name of the module containing Go std packages.
	stdModule = "std"
)

// generateDB generates the file-based vuln DB in the directory jsonDir.
func generateDB(ctx context.Context, txtarData []byte, jsonDir string, indent bool) error {
	archive := txtar.Parse(txtarData)

	entries, err := generateEntries(ctx, archive)
	if err != nil {
		return err
	}
	return writeEntriesByID(filepath.Join(jsonDir, idDirectory), entries, indent)
}

func generateEntries(_ context.Context, archive *txtar.Archive) ([]osv.Entry, error) {
	now := time.Now()
	var entries []osv.Entry
	for _, f := range archive.Files {
		if !strings.HasSuffix(f.Name, ".yaml") {
			continue
		}
		r, err := readReport(bytes.NewReader(f.Data))
		if err != nil {
			return nil, err
		}
		name := strings.TrimSuffix(filepath.Base(f.Name), filepath.Ext(f.Name))
		linkName := fmt.Sprintf("%s%s", dbURL, name)
		entry := generateOSVEntry(name, linkName, now, *r)
		entries = append(entries, entry)
	}
	return entries, nil
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
	for _, e := range entries {
		outPath := filepath.Join(idDir, e.ID+".json")
		if err := writeJSON(outPath, e, indent); err != nil {
			return err
		}
	}
	return nil
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
func generateOSVEntry(id, url string, lastModified time.Time, r Report) osv.Entry {
	entry := osv.Entry{
		ID:               id,
		Published:        r.Published,
		Modified:         lastModified,
		Withdrawn:        r.Withdrawn,
		Summary:          r.Summary,
		Details:          r.Description,
		DatabaseSpecific: &osv.DatabaseSpecific{URL: url},
	}

	moduleMap := make(map[string]bool)
	for _, m := range r.Modules {
		switch m.Module {
		case stdModule:
			moduleMap[osv.GoStdModulePath] = true
		case cmdModule:
			moduleMap[osv.GoCmdModulePath] = true
		default:
			moduleMap[m.Module] = true
		}
		entry.Affected = append(entry.Affected, toAffected(m))
	}
	for _, ref := range r.References {
		entry.References = append(entry.References, osv.Reference{
			Type: ref.Type,
			URL:  ref.URL,
		})
	}
	return entry
}

func AffectedRanges(versions []VersionRange) []osv.Range {
	a := osv.Range{Type: osv.RangeTypeSemver}
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
	return []osv.Range{a}
}

func toOSVPackages(pkgs []*Package) (imps []osv.Package) {
	for _, p := range pkgs {
		syms := append([]string{}, p.Symbols...)
		syms = append(syms, p.DerivedSymbols...)
		sort.Strings(syms)
		imps = append(imps, osv.Package{
			Path:    p.Package,
			GOOS:    p.GOOS,
			GOARCH:  p.GOARCH,
			Symbols: syms,
		})
	}
	return imps
}

func toAffected(m *Module) osv.Affected {
	name := m.Module
	switch name {
	case stdModule:
		name = osv.GoStdModulePath
	case cmdModule:
		name = osv.GoCmdModulePath
	}
	return osv.Affected{
		Module: osv.Module{
			Path:      name,
			Ecosystem: osv.GoEcosystem,
		},
		Ranges: AffectedRanges(m.Versions),
		EcosystemSpecific: osv.EcosystemSpecific{
			Packages: toOSVPackages(m.Packages),
		},
	}
}
