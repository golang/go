// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package vulntest

import (
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"golang.org/x/mod/semver"
	"golang.org/x/tools/gopls/internal/vulncheck/osv"
	"gopkg.in/yaml.v3"
)

//
// The following was selectively copied from golang.org/x/vulndb/internal/report
//

// readReport reads a Report in YAML format.
func readReport(in io.Reader) (*Report, error) {
	d := yaml.NewDecoder(in)
	// Require that all fields in the file are in the struct.
	// This corresponds to v2's UnmarshalStrict.
	d.KnownFields(true)
	var r Report
	if err := d.Decode(&r); err != nil {
		return nil, fmt.Errorf("yaml.Decode: %v", err)
	}
	return &r, nil
}

// Report represents a vulnerability report in the vulndb.
// See https://go.googlesource.com/vulndb/+/refs/heads/master/doc/format.md
type Report struct {
	ID string `yaml:",omitempty"`

	Modules []*Module `yaml:",omitempty"`

	// Summary is a short phrase describing the vulnerability.
	Summary string `yaml:",omitempty"`

	// Description is the CVE description from an existing CVE. If we are
	// assigning a CVE ID ourselves, use CVEMetadata.Description instead.
	Description string     `yaml:",omitempty"`
	Published   time.Time  `yaml:",omitempty"`
	Withdrawn   *time.Time `yaml:",omitempty"`

	References []*Reference `yaml:",omitempty"`
}

// Write writes r to filename in YAML format.
func (r *Report) Write(filename string) (err error) {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	err = r.encode(f)
	err2 := f.Close()
	if err == nil {
		err = err2
	}
	return err
}

// ToString encodes r to a YAML string.
func (r *Report) ToString() (string, error) {
	var b strings.Builder
	if err := r.encode(&b); err != nil {
		return "", err
	}
	return b.String(), nil
}

func (r *Report) encode(w io.Writer) error {
	e := yaml.NewEncoder(w)
	defer e.Close()
	e.SetIndent(4)
	return e.Encode(r)
}

type VersionRange struct {
	Introduced Version `yaml:"introduced,omitempty"`
	Fixed      Version `yaml:"fixed,omitempty"`
}

type Module struct {
	Module   string         `yaml:",omitempty"`
	Versions []VersionRange `yaml:",omitempty"`
	Packages []*Package     `yaml:",omitempty"`
}

type Package struct {
	Package string   `yaml:",omitempty"`
	GOOS    []string `yaml:"goos,omitempty"`
	GOARCH  []string `yaml:"goarch,omitempty"`
	// Symbols originally identified as vulnerable.
	Symbols []string `yaml:",omitempty"`
	// Additional vulnerable symbols, computed from Symbols via static analysis
	// or other technique.
	DerivedSymbols []string `yaml:"derived_symbols,omitempty"`
}

// Version is an SemVer 2.0.0 semantic version with no leading "v" prefix,
// as used by OSV.
type Version string

// V returns the version with a "v" prefix.
func (v Version) V() string {
	return "v" + string(v)
}

// IsValid reports whether v is a valid semantic version string.
func (v Version) IsValid() bool {
	return semver.IsValid(v.V())
}

// Before reports whether v < v2.
func (v Version) Before(v2 Version) bool {
	return semver.Compare(v.V(), v2.V()) < 0
}

// Canonical returns the canonical formatting of the version.
func (v Version) Canonical() string {
	return strings.TrimPrefix(semver.Canonical(v.V()), "v")
}

// Reference type is a reference (link) type.
type ReferenceType string

const (
	ReferenceTypeAdvisory = ReferenceType("ADVISORY")
	ReferenceTypeArticle  = ReferenceType("ARTICLE")
	ReferenceTypeReport   = ReferenceType("REPORT")
	ReferenceTypeFix      = ReferenceType("FIX")
	ReferenceTypePackage  = ReferenceType("PACKAGE")
	ReferenceTypeEvidence = ReferenceType("EVIDENCE")
	ReferenceTypeWeb      = ReferenceType("WEB")
)

// ReferenceTypes is the set of reference types defined in OSV.
var ReferenceTypes = []ReferenceType{
	ReferenceTypeAdvisory,
	ReferenceTypeArticle,
	ReferenceTypeReport,
	ReferenceTypeFix,
	ReferenceTypePackage,
	ReferenceTypeEvidence,
	ReferenceTypeWeb,
}

// A Reference is a link to some external resource.
//
// For ease of typing, References are represented in the YAML as a
// single-element mapping of type to URL.
type Reference osv.Reference

func (r *Reference) MarshalYAML() (interface{}, error) {
	return map[string]string{
		strings.ToLower(string(r.Type)): r.URL,
	}, nil
}

func (r *Reference) UnmarshalYAML(n *yaml.Node) (err error) {
	if n.Kind != yaml.MappingNode || len(n.Content) != 2 || n.Content[0].Kind != yaml.ScalarNode || n.Content[1].Kind != yaml.ScalarNode {
		return &yaml.TypeError{Errors: []string{
			fmt.Sprintf("line %d: report.Reference must contain a mapping with one value", n.Line),
		}}
	}
	r.Type = osv.ReferenceType(strings.ToUpper(n.Content[0].Value))
	r.URL = n.Content[1].Value
	return nil
}
