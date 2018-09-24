package analysis

import (
	"fmt"
	"unicode"
)

// Validate reports an error if any of the analyzers are misconfigured.
// Checks include:
// - that the name is a valid identifier;
// - that analysis names are unique;
// - that the Requires graph is acylic.
func Validate(analyzers []*Analyzer) error {
	names := make(map[string]bool)

	// Traverse the Requires graph, depth first.
	color := make(map[*Analyzer]uint8) // 0=white 1=grey 2=black
	var visit func(a *Analyzer) error
	visit = func(a *Analyzer) error {
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
	for _, a := range analyzers {
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
