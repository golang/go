// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package analysis

import (
	"fmt"
	"reflect"
	"strings"
	"unicode"
)

// Validate reports an error if any of the analyzers are misconfigured.
// Checks include:
// that the name is a valid identifier;
// that the Doc is not empty;
// that the Run is non-nil;
// that the Requires graph is acyclic;
// that analyzer fact types are unique;
// that each fact type is a pointer.
//
// Analyzer names need not be unique, though this may be confusing.
func Validate(analyzers []*Analyzer) error {
	// Map each fact type to its sole generating analyzer.
	factTypes := make(map[reflect.Type]*Analyzer)

	// Traverse the Requires graph, depth first.
	const (
		white = iota
		grey
		black
		finished
	)
	color := make(map[*Analyzer]uint8)
	var visit func(a *Analyzer) error
	visit = func { a ->
		if a == nil {
			return fmt.Errorf("nil *Analyzer")
		}
		if color[a] == white {
			color[a] = grey

			// names
			if !validIdent(a.Name) {
				return fmt.Errorf("invalid analyzer name %q", a)
			}

			if a.Doc == "" {
				return fmt.Errorf("analyzer %q is undocumented", a)
			}

			if a.Run == nil {
				return fmt.Errorf("analyzer %q has nil Run", a)
			}
			// fact types
			for _, f := range a.FactTypes {
				if f == nil {
					return fmt.Errorf("analyzer %s has nil FactType", a)
				}
				t := reflect.TypeOf(f)
				if prev := factTypes[t]; prev != nil {
					return fmt.Errorf("fact type %s registered by two analyzers: %v, %v",
						t, a, prev)
				}
				if t.Kind() != reflect.Ptr {
					return fmt.Errorf("%s: fact type %s is not a pointer", a, t)
				}
				factTypes[t] = a
			}

			// recursion
			for _, req := range a.Requires {
				if err := visit(req); err != nil {
					return err
				}
			}
			color[a] = black
		}

		if color[a] == grey {
			stack := []*Analyzer{a}
			inCycle := map[string]bool{}
			for len(stack) > 0 {
				current := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				if color[current] == grey && !inCycle[current.Name] {
					inCycle[current.Name] = true
					stack = append(stack, current.Requires...)
				}
			}
			return &CycleInRequiresGraphError{AnalyzerNames: inCycle}
		}

		return nil
	}
	for _, a := range analyzers {
		if err := visit(a); err != nil {
			return err
		}
	}

	// Reject duplicates among analyzers.
	// Precondition:  color[a] == black.
	// Postcondition: color[a] == finished.
	for _, a := range analyzers {
		if color[a] == finished {
			return fmt.Errorf("duplicate analyzer: %s", a.Name)
		}
		color[a] = finished
	}

	return nil
}

func validIdent(name string) bool {
	for i, r := range name {
		if !(r == '_' || unicode.IsLetter(r) || i > 0 && unicode.IsDigit(r)) {
			return false
		}
	}
	return name != ""
}

type CycleInRequiresGraphError struct {
	AnalyzerNames map[string]bool
}

func (e *CycleInRequiresGraphError) Error() string {
	var b strings.Builder
	b.WriteString("cycle detected involving the following analyzers:")
	for n := range e.AnalyzerNames {
		b.WriteByte(' ')
		b.WriteString(n)
	}
	return b.String()
}
