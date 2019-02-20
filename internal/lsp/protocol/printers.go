// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains formatting functions for types that
// are commonly printed in debugging information.
// They are separated from their types and gathered here as
// they are hand written and not generated from the spec.
// They should not be relied on for programmatic use (their
// results should never be parsed for instance) but are meant
// for temporary debugging and error messages.

package protocol

import (
	"fmt"
)

func (s DiagnosticSeverity) Format(f fmt.State, c rune) {
	switch s {
	case SeverityError:
		fmt.Fprint(f, "Error")
	case SeverityWarning:
		fmt.Fprint(f, "Warning")
	case SeverityInformation:
		fmt.Fprint(f, "Information")
	case SeverityHint:
		fmt.Fprint(f, "Hint")
	}
}

func (k CompletionItemKind) Format(f fmt.State, c rune) {
	switch k {
	case StructCompletion:
		fmt.Fprintf(f, "struct")
	case FunctionCompletion:
		fmt.Fprintf(f, "func")
	case VariableCompletion:
		fmt.Fprintf(f, "var")
	case TypeParameterCompletion:
		fmt.Fprintf(f, "type")
	case FieldCompletion:
		fmt.Fprintf(f, "field")
	case InterfaceCompletion:
		fmt.Fprintf(f, "interface")
	case ConstantCompletion:
		fmt.Fprintf(f, "const")
	case MethodCompletion:
		fmt.Fprintf(f, "method")
	case ModuleCompletion:
		fmt.Fprintf(f, "package")
	}
}
