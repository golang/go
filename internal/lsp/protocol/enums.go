// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocol

import (
	"fmt"
)

var (
	namesTextDocumentSyncKind   [int(Incremental) + 1]string
	namesInitializeError        [int(UnknownProtocolVersion) + 1]string
	namesMessageType            [int(Log) + 1]string
	namesFileChangeType         [int(Deleted) + 1]string
	namesWatchKind              [int(WatchDelete) + 1]string
	namesCompletionTriggerKind  [int(TriggerForIncompleteCompletions) + 1]string
	namesDiagnosticSeverity     [int(SeverityHint) + 1]string
	namesDiagnosticTag          [int(Unnecessary) + 1]string
	namesCompletionItemKind     [int(TypeParameterCompletion) + 1]string
	namesInsertTextFormat       [int(SnippetTextFormat) + 1]string
	namesDocumentHighlightKind  [int(Write) + 1]string
	namesSymbolKind             [int(TypeParameter) + 1]string
	namesTextDocumentSaveReason [int(FocusOut) + 1]string
)

func init() {
	namesTextDocumentSyncKind[int(None)] = "None"
	namesTextDocumentSyncKind[int(Full)] = "Full"
	namesTextDocumentSyncKind[int(Incremental)] = "Incremental"

	namesInitializeError[int(UnknownProtocolVersion)] = "UnknownProtocolVersion"

	namesMessageType[int(Error)] = "Error"
	namesMessageType[int(Warning)] = "Warning"
	namesMessageType[int(Info)] = "Info"
	namesMessageType[int(Log)] = "Log"

	namesFileChangeType[int(Created)] = "Created"
	namesFileChangeType[int(Changed)] = "Changed"
	namesFileChangeType[int(Deleted)] = "Deleted"

	namesWatchKind[int(WatchCreate)] = "WatchCreate"
	namesWatchKind[int(WatchChange)] = "WatchChange"
	namesWatchKind[int(WatchDelete)] = "WatchDelete"

	namesCompletionTriggerKind[int(Invoked)] = "Invoked"
	namesCompletionTriggerKind[int(TriggerCharacter)] = "TriggerCharacter"
	namesCompletionTriggerKind[int(TriggerForIncompleteCompletions)] = "TriggerForIncompleteCompletions"

	namesDiagnosticSeverity[int(SeverityError)] = "Error"
	namesDiagnosticSeverity[int(SeverityWarning)] = "Warning"
	namesDiagnosticSeverity[int(SeverityInformation)] = "Information"
	namesDiagnosticSeverity[int(SeverityHint)] = "Hint"

	namesDiagnosticTag[int(Unnecessary)] = "Unnecessary"

	namesCompletionItemKind[int(TextCompletion)] = "text"
	namesCompletionItemKind[int(MethodCompletion)] = "method"
	namesCompletionItemKind[int(FunctionCompletion)] = "func"
	namesCompletionItemKind[int(ConstructorCompletion)] = "constructor"
	namesCompletionItemKind[int(FieldCompletion)] = "field"
	namesCompletionItemKind[int(VariableCompletion)] = "var"
	namesCompletionItemKind[int(ClassCompletion)] = "class"
	namesCompletionItemKind[int(InterfaceCompletion)] = "interface"
	namesCompletionItemKind[int(ModuleCompletion)] = "package"
	namesCompletionItemKind[int(PropertyCompletion)] = "property"
	namesCompletionItemKind[int(UnitCompletion)] = "unit"
	namesCompletionItemKind[int(ValueCompletion)] = "value"
	namesCompletionItemKind[int(EnumCompletion)] = "enum"
	namesCompletionItemKind[int(KeywordCompletion)] = "keyword"
	namesCompletionItemKind[int(SnippetCompletion)] = "snippet"
	namesCompletionItemKind[int(ColorCompletion)] = "color"
	namesCompletionItemKind[int(FileCompletion)] = "file"
	namesCompletionItemKind[int(ReferenceCompletion)] = "reference"
	namesCompletionItemKind[int(FolderCompletion)] = "folder"
	namesCompletionItemKind[int(EnumMemberCompletion)] = "enumMember"
	namesCompletionItemKind[int(ConstantCompletion)] = "const"
	namesCompletionItemKind[int(StructCompletion)] = "struct"
	namesCompletionItemKind[int(EventCompletion)] = "event"
	namesCompletionItemKind[int(OperatorCompletion)] = "operator"
	namesCompletionItemKind[int(TypeParameterCompletion)] = "type"

	namesInsertTextFormat[int(PlainTextTextFormat)] = "PlainText"
	namesInsertTextFormat[int(SnippetTextFormat)] = "Snippet"

	namesDocumentHighlightKind[int(Text)] = "Text"
	namesDocumentHighlightKind[int(Read)] = "Read"
	namesDocumentHighlightKind[int(Write)] = "Write"

	namesSymbolKind[int(File)] = "File"
	namesSymbolKind[int(Module)] = "Module"
	namesSymbolKind[int(Namespace)] = "Namespace"
	namesSymbolKind[int(Package)] = "Package"
	namesSymbolKind[int(Class)] = "Class"
	namesSymbolKind[int(Method)] = "Method"
	namesSymbolKind[int(Property)] = "Property"
	namesSymbolKind[int(Field)] = "Field"
	namesSymbolKind[int(Constructor)] = "Constructor"
	namesSymbolKind[int(Enum)] = "Enum"
	namesSymbolKind[int(Interface)] = "Interface"
	namesSymbolKind[int(Function)] = "Function"
	namesSymbolKind[int(Variable)] = "Variable"
	namesSymbolKind[int(Constant)] = "Constant"
	namesSymbolKind[int(String)] = "String"
	namesSymbolKind[int(Number)] = "Number"
	namesSymbolKind[int(Boolean)] = "Boolean"
	namesSymbolKind[int(Array)] = "Array"
	namesSymbolKind[int(Object)] = "Object"
	namesSymbolKind[int(Key)] = "Key"
	namesSymbolKind[int(Null)] = "Null"
	namesSymbolKind[int(EnumMember)] = "EnumMember"
	namesSymbolKind[int(Struct)] = "Struct"
	namesSymbolKind[int(Event)] = "Event"
	namesSymbolKind[int(Operator)] = "Operator"
	namesSymbolKind[int(TypeParameter)] = "TypeParameter"

	namesTextDocumentSaveReason[int(Manual)] = "Manual"
	namesTextDocumentSaveReason[int(AfterDelay)] = "AfterDelay"
	namesTextDocumentSaveReason[int(FocusOut)] = "FocusOut"
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

func (e TextDocumentSyncKind) Format(f fmt.State, c rune) {
	formatEnum(f, c, int(e), namesTextDocumentSyncKind[:], "TextDocumentSyncKind")
}

func ParseTextDocumentSyncKind(s string) TextDocumentSyncKind {
	return TextDocumentSyncKind(parseEnum(s, namesTextDocumentSyncKind[:]))
}

func (e InitializeError) Format(f fmt.State, c rune) {
	formatEnum(f, c, int(e), namesInitializeError[:], "InitializeError")
}

func ParseInitializeError(s string) InitializeError {
	return InitializeError(parseEnum(s, namesInitializeError[:]))
}

func (e MessageType) Format(f fmt.State, c rune) {
	formatEnum(f, c, int(e), namesMessageType[:], "MessageType")
}

func ParseMessageType(s string) MessageType {
	return MessageType(parseEnum(s, namesMessageType[:]))
}

func (e FileChangeType) Format(f fmt.State, c rune) {
	formatEnum(f, c, int(e), namesFileChangeType[:], "FileChangeType")
}

func ParseFileChangeType(s string) FileChangeType {
	return FileChangeType(parseEnum(s, namesFileChangeType[:]))
}

func (e WatchKind) Format(f fmt.State, c rune) {
	formatEnum(f, c, int(e), namesWatchKind[:], "WatchKind")
}

func ParseWatchKind(s string) WatchKind {
	return WatchKind(parseEnum(s, namesWatchKind[:]))
}

func (e CompletionTriggerKind) Format(f fmt.State, c rune) {
	formatEnum(f, c, int(e), namesCompletionTriggerKind[:], "CompletionTriggerKind")
}

func ParseCompletionTriggerKind(s string) CompletionTriggerKind {
	return CompletionTriggerKind(parseEnum(s, namesCompletionTriggerKind[:]))
}

func (e DiagnosticSeverity) Format(f fmt.State, c rune) {
	formatEnum(f, c, int(e), namesDiagnosticSeverity[:], "DiagnosticSeverity")
}

func ParseDiagnosticSeverity(s string) DiagnosticSeverity {
	return DiagnosticSeverity(parseEnum(s, namesDiagnosticSeverity[:]))
}

func (e DiagnosticTag) Format(f fmt.State, c rune) {
	formatEnum(f, c, int(e), namesDiagnosticTag[:], "DiagnosticTag")
}

func ParseDiagnosticTag(s string) DiagnosticTag {
	return DiagnosticTag(parseEnum(s, namesDiagnosticTag[:]))
}

func (e CompletionItemKind) Format(f fmt.State, c rune) {
	formatEnum(f, c, int(e), namesCompletionItemKind[:], "CompletionItemKind")
}

func ParseCompletionItemKind(s string) CompletionItemKind {
	return CompletionItemKind(parseEnum(s, namesCompletionItemKind[:]))
}

func (e InsertTextFormat) Format(f fmt.State, c rune) {
	formatEnum(f, c, int(e), namesInsertTextFormat[:], "InsertTextFormat")
}

func ParseInsertTextFormat(s string) InsertTextFormat {
	return InsertTextFormat(parseEnum(s, namesInsertTextFormat[:]))
}

func (e DocumentHighlightKind) Format(f fmt.State, c rune) {
	formatEnum(f, c, int(e), namesDocumentHighlightKind[:], "DocumentHighlightKind")
}

func ParseDocumentHighlightKind(s string) DocumentHighlightKind {
	return DocumentHighlightKind(parseEnum(s, namesDocumentHighlightKind[:]))
}

func (e SymbolKind) Format(f fmt.State, c rune) {
	formatEnum(f, c, int(e), namesSymbolKind[:], "SymbolKind")
}

func ParseSymbolKind(s string) SymbolKind {
	return SymbolKind(parseEnum(s, namesSymbolKind[:]))
}

func (e TextDocumentSaveReason) Format(f fmt.State, c rune) {
	formatEnum(f, c, int(e), namesTextDocumentSaveReason[:], "TextDocumentSaveReason")
}

func ParseTextDocumentSaveReason(s string) TextDocumentSaveReason {
	return TextDocumentSaveReason(parseEnum(s, namesTextDocumentSaveReason[:]))
}
