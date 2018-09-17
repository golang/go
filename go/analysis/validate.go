package analysis

import (
	"fmt"
	"reflect"
	"unicode"
)

// Validate reports an error if any of the analyses are misconfigured.
// Checks include:
// - that the name is a valid identifier;
// - that analysis names are unique;
// - that the Requires graph is acylic;
// - that analyses' lemma and output types are unique.
// - that each lemma type is a pointer.
func Validate(analyses []*Analysis) error {
	names := make(map[string]bool)

	// Map each lemma/output type to its sole generating analysis.
	lemmaTypes := make(map[reflect.Type]*Analysis)
	outputTypes := make(map[reflect.Type]*Analysis)

	// Traverse the Requires graph, depth first.
	color := make(map[*Analysis]uint8) // 0=white 1=grey 2=black
	var visit func(a *Analysis) error
	visit = func(a *Analysis) error {
		if a == nil {
			return fmt.Errorf("nil *Analysis")
		}
		if color[a] == 0 { // white
			color[a] = 1 // grey

			// names
			if !validIdent(a.Name) {
				return fmt.Errorf("invalid analysis name %q", a)
			}
			if names[a.Name] {
				return fmt.Errorf("duplicate analysis name %q", a)
			}
			names[a.Name] = true

			if a.Doc == "" {
				return fmt.Errorf("analysis %q is undocumented", a)
			}

			// lemma types
			for _, t := range a.LemmaTypes {
				if t == nil {
					return fmt.Errorf("analysis %s has nil LemmaType", a)
				}
				if prev := lemmaTypes[t]; prev != nil {
					return fmt.Errorf("lemma type %s registered by two analyses: %v, %v",
						t, a, prev)
				}
				if t.Kind() != reflect.Ptr {
					return fmt.Errorf("%s: lemma type %s is not a pointer", a, t)
				}
				lemmaTypes[t] = a
			}

			// output types
			if a.OutputType != nil {
				if prev := outputTypes[a.OutputType]; prev != nil {
					return fmt.Errorf("output type %s registered by two analyses: %v, %v",
						a.OutputType, a, prev)
				}
				outputTypes[a.OutputType] = a
			}

			// recursion
			for i, req := range a.Requires {
				if err := visit(req); err != nil {
					return fmt.Errorf("%s.Requires[%d]: %v", a.Name, i, err)
				}
			}
			color[a] = 2 // black
		}

		return nil
	}
	for _, a := range analyses {
		if err := visit(a); err != nil {
			return err
		}
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
