// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import "fmt"

var (
	namesSymbolKind         [int(FieldSymbol) + 1]string
	namesDiagnosticSeverity [int(SeverityError) + 1]string
	namesCompletionItemKind [int(PackageCompletionItem) + 1]string
)

func init() {
	namesSymbolKind[PackageSymbol] = "Package"
	namesSymbolKind[StructSymbol] = "Struct"
	namesSymbolKind[VariableSymbol] = "Variable"
	namesSymbolKind[ConstantSymbol] = "Constant"
	namesSymbolKind[FunctionSymbol] = "Function"
	namesSymbolKind[MethodSymbol] = "Method"
	namesSymbolKind[InterfaceSymbol] = "Interface"
	namesSymbolKind[NumberSymbol] = "Number"
	namesSymbolKind[StringSymbol] = "String"
	namesSymbolKind[BooleanSymbol] = "Boolean"
	namesSymbolKind[FieldSymbol] = "Field"

	namesDiagnosticSeverity[SeverityWarning] = "Warning"
	namesDiagnosticSeverity[SeverityError] = "Error"

	namesCompletionItemKind[Unknown] = "Unknown"
	namesCompletionItemKind[InterfaceCompletionItem] = "interface"
	namesCompletionItemKind[StructCompletionItem] = "struct"
	namesCompletionItemKind[TypeCompletionItem] = "type"
	namesCompletionItemKind[ConstantCompletionItem] = "const"
	namesCompletionItemKind[FieldCompletionItem] = "field"
	namesCompletionItemKind[ParameterCompletionItem] = "parameter"
	namesCompletionItemKind[VariableCompletionItem] = "var"
	namesCompletionItemKind[FunctionCompletionItem] = "func"
	namesCompletionItemKind[MethodCompletionItem] = "method"
	namesCompletionItemKind[PackageCompletionItem] = "package"
}

func formatEnum(f fmt.State, c rune, i int, names []string, unknown string) {
	s := ""
	if i >= 0 && i < len(names) {
		s = names[i]
	}
	if s != "" {
		fmt.Fprint(f, s)
	} else {
		fmt.Fprintf(f, "%s(%d)", unknown, i)
	}
}

func parseEnum(s string, names []string) int {
	for i, name := range names {
		if s == name {
			return i
		}
	}
	return 0
}

func (e SymbolKind) Format(f fmt.State, c rune) {
	formatEnum(f, c, int(e), namesSymbolKind[:], "SymbolKind")
}

func ParseSymbolKind(s string) SymbolKind {
	return SymbolKind(parseEnum(s, namesSymbolKind[:]))
}

func (e DiagnosticSeverity) Format(f fmt.State, c rune) {
	formatEnum(f, c, int(e), namesDiagnosticSeverity[:], "DiagnosticSeverity")
}

func ParseDiagnosticSeverity(s string) DiagnosticSeverity {
	return DiagnosticSeverity(parseEnum(s, namesDiagnosticSeverity[:]))
}

func (e CompletionItemKind) Format(f fmt.State, c rune) {
	formatEnum(f, c, int(e), namesCompletionItemKind[:], "CompletionItemKind")
}

func ParseCompletionItemKind(s string) CompletionItemKind {
	return CompletionItemKind(parseEnum(s, namesCompletionItemKind[:]))
}
