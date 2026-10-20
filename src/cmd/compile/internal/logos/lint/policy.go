// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// PolicyVersion is the version of the policy file the go command writes
// for -logos.cfg.
const PolicyVersion = 1

// Thresholds holds the [lints.thresholds] values.
type Thresholds struct {
	MaxNestingDepth        int
	MaxCognitiveComplexity int
	MaxFunctionLines       int
	MaxFunctionParams      int
	MaxComptimeASTNodes    int
	MaxCTFESteps           int
}

// Policy is the resolved logos.toml [lints] table: one severity per group
// and the six thresholds.
type Policy struct {
	groups     map[Group]Severity
	Thresholds Thresholds
}

// DefaultPolicy returns the policy of a module with no [lints] table.
func DefaultPolicy() *Policy {
	return &Policy{
		groups: make(map[Group]Severity),
		Thresholds: Thresholds{
			MaxNestingDepth:        3,
			MaxCognitiveComplexity: 15,
			MaxFunctionLines:       60,
			MaxFunctionParams:      4,
			MaxComptimeASTNodes:    4096,
			MaxCTFESteps:           100000,
		},
	}
}

// policyFile is the JSON the go command writes for -logos.cfg. Absent
// groups and thresholds keep their defaults.
type policyFile struct {
	Version    int               `json:"version"`
	Groups     map[string]string `json:"groups"`
	Thresholds map[string]int    `json:"thresholds"`
}

// ParsePolicy decodes the -logos.cfg JSON. It rejects unknown fields,
// unknown groups, unknown thresholds, non-positive thresholds, and a
// forbid group set below forbid. It uses encoding/json, not
// encoding/json/v2: the bootstrap toolchain builds cmd/compile, and v2 is an
// experiment there.
func ParsePolicy(data []byte) (*Policy, error) {
	var f policyFile
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&f); err != nil {
		return nil, fmt.Errorf("logos policy: %v", err)
	}
	if f.Version != PolicyVersion {
		return nil, fmt.Errorf("logos policy: unsupported version %d", f.Version)
	}
	p := DefaultPolicy()
	for name, level := range f.Groups {
		if err := p.setGroup(name, level); err != nil {
			return nil, err
		}
	}
	for name, v := range f.Thresholds {
		if err := p.setThreshold(name, v); err != nil {
			return nil, err
		}
	}
	return p, nil
}

// setGroup stores one group severity.
func (p *Policy) setGroup(name, level string) error {
	g := Group(name)
	if !knownGroup(g) {
		return fmt.Errorf("logos policy: unknown group %q", name)
	}
	s, ok := ParseSeverity(level)
	if !ok {
		return fmt.Errorf("logos policy: invalid severity %q for group %s", level, name)
	}
	if DefaultSeverity(g) == Forbid && s != Forbid {
		return fmt.Errorf("logos policy: group %s cannot be lowered below forbid", name)
	}
	p.groups[g] = s
	return nil
}

// setThreshold stores one threshold.
func (p *Policy) setThreshold(name string, v int) error {
	if v < 1 {
		return fmt.Errorf("logos policy: threshold %s must be positive", name)
	}
	field := p.thresholdField(name)
	if field == nil {
		return fmt.Errorf("logos policy: unknown threshold %q", name)
	}
	*field = v
	return nil
}

// thresholdField returns the Thresholds field for a logos.toml key, or nil.
func (p *Policy) thresholdField(name string) *int {
	switch name {
	case "max_nesting_depth":
		return &p.Thresholds.MaxNestingDepth
	case "max_cognitive_complexity":
		return &p.Thresholds.MaxCognitiveComplexity
	case "max_function_lines":
		return &p.Thresholds.MaxFunctionLines
	case "max_function_params":
		return &p.Thresholds.MaxFunctionParams
	case "max_comptime_ast_nodes":
		return &p.Thresholds.MaxComptimeASTNodes
	case "max_ctfe_steps":
		return &p.Thresholds.MaxCTFESteps
	}
	return nil
}

// knownGroup reports whether g is one of the nine lint groups.
func knownGroup(g Group) bool {
	for _, k := range Groups {
		if k == g {
			return true
		}
	}
	return false
}

// GroupOf returns the group part of a rule ID, or "" for an ID with no "::".
func GroupOf(id string) Group {
	group, _, ok := strings.Cut(id, "::")
	if !ok {
		return ""
	}
	return Group(group)
}

// Level returns the severity rule id reports at in a package of kind k.
// A Go package never runs at forbid: the Go cap lowers it to Deny, so a
// //logos:allow directive can silence a Go false positive.
func (p *Policy) Level(id string, k Kind) Severity {
	g := GroupOf(id)
	s, ok := p.groups[g]
	if !ok {
		s = defaultLevel(id, g)
	}
	if k == KindGo && s == Forbid {
		return Deny
	}
	return s
}

// defaultLevel returns the severity of rule id when the policy names none
// for its group g: the Default of the registered rule, else the default
// severity of the group.
func defaultLevel(id string, g Group) Severity {
	if r, ok := Lookup(id); ok && r.Default != Allow {
		return r.Default
	}
	return DefaultSeverity(g)
}
