package analysis

import (
	"fmt"
	"reflect"
	"unicode"
)

// Validate reports an error if any of the analyzers are misconfigured.
// Checks include:
// that the name is a valid identifier;
// that analyzer names are unique;
// that the Requires graph is acylic;
// that analyzer fact types are unique;
// that each fact type is a pointer.
func Validate(analyzers []*Analyzer) error {
	names := make(map[string]bool)

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
	visit = func(a *Analyzer) error {
		if a == nil {
			return fmt.Errorf("nil *Analyzer")
		}
		if color[a] == white {
			color[a] = grey

			// names
			if !validIdent(a.Name) {
				return fmt.Errorf("invalid analyzer name %q", a)
			}
			if names[a.Name] {
				return fmt.Errorf("duplicate analyzer name %q", a)
			}
			names[a.Name] = true

			if a.Doc == "" {
				return fmt.Errorf("analyzer %q is undocumented", a)
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
			for i, req := range a.Requires {
				if err := visit(req); err != nil {
					return fmt.Errorf("%s.Requires[%d]: %v", a.Name, i, err)
				}
			}
			color[a] = black
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
