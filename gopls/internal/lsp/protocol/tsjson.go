// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Code generated for LSP. DO NOT EDIT.

package protocol

// Code generated from protocol/metaModel.json at ref release/protocol/3.17.4-next.2 (hash 184c8a7f010d335582f24337fe182baa6f2fccdd).
// https://github.com/microsoft/vscode-languageserver-node/blob/release/protocol/3.17.4-next.2/protocol/metaModel.json
// LSP metaData.version = 3.17.0.

import "encoding/json"

import "fmt"

// UnmarshalError indicates that a JSON value did not conform to
// one of the expected cases of an LSP union type.
type UnmarshalError struct {
	msg string
}

func (e UnmarshalError) Error() string {
	return e.msg
}

// from line 4964
func (t OrFEditRangePItemDefaults) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case FEditRangePItemDefaults:
		return json.Marshal(x)
	case Range:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [FEditRangePItemDefaults Range]", t)
}

func (t *OrFEditRangePItemDefaults) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 FEditRangePItemDefaults
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 Range
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [FEditRangePItemDefaults Range]"}
}

// from line 10165
func (t OrFNotebookPNotebookSelector) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case NotebookDocumentFilter:
		return json.Marshal(x)
	case string:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [NotebookDocumentFilter string]", t)
}

func (t *OrFNotebookPNotebookSelector) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 NotebookDocumentFilter
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 string
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [NotebookDocumentFilter string]"}
}

// from line 5715
func (t OrPLocation_workspace_symbol) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case Location:
		return json.Marshal(x)
	case PLocationMsg_workspace_symbol:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [Location PLocationMsg_workspace_symbol]", t)
}

func (t *OrPLocation_workspace_symbol) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 Location
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 PLocationMsg_workspace_symbol
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [Location PLocationMsg_workspace_symbol]"}
}

// from line 4358
func (t OrPSection_workspace_didChangeConfiguration) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case []string:
		return json.Marshal(x)
	case string:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [[]string string]", t)
}

func (t *OrPSection_workspace_didChangeConfiguration) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 []string
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 string
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [[]string string]"}
}

// from line 7311
func (t OrPTooltipPLabel) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case MarkupContent:
		return json.Marshal(x)
	case string:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [MarkupContent string]", t)
}

func (t *OrPTooltipPLabel) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 MarkupContent
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 string
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [MarkupContent string]"}
}

// from line 3772
func (t OrPTooltip_textDocument_inlayHint) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case MarkupContent:
		return json.Marshal(x)
	case string:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [MarkupContent string]", t)
}

func (t *OrPTooltip_textDocument_inlayHint) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 MarkupContent
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 string
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [MarkupContent string]"}
}

// from line 6420
func (t Or_CancelParams_id) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case int32:
		return json.Marshal(x)
	case string:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [int32 string]", t)
}

func (t *Or_CancelParams_id) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 int32
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 string
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [int32 string]"}
}

// from line 4777
func (t Or_CompletionItem_documentation) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case MarkupContent:
		return json.Marshal(x)
	case string:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [MarkupContent string]", t)
}

func (t *Or_CompletionItem_documentation) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 MarkupContent
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 string
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [MarkupContent string]"}
}

// from line 4860
func (t Or_CompletionItem_textEdit) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case InsertReplaceEdit:
		return json.Marshal(x)
	case TextEdit:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [InsertReplaceEdit TextEdit]", t)
}

func (t *Or_CompletionItem_textEdit) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 InsertReplaceEdit
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 TextEdit
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [InsertReplaceEdit TextEdit]"}
}

// from line 14168
func (t Or_Definition) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case Location:
		return json.Marshal(x)
	case []Location:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [Location []Location]", t)
}

func (t *Or_Definition) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 Location
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 []Location
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [Location []Location]"}
}

// from line 8865
func (t Or_Diagnostic_code) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case int32:
		return json.Marshal(x)
	case string:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [int32 string]", t)
}

func (t *Or_Diagnostic_code) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 int32
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 string
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [int32 string]"}
}

// from line 14300
func (t Or_DocumentDiagnosticReport) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case RelatedFullDocumentDiagnosticReport:
		return json.Marshal(x)
	case RelatedUnchangedDocumentDiagnosticReport:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [RelatedFullDocumentDiagnosticReport RelatedUnchangedDocumentDiagnosticReport]", t)
}

func (t *Or_DocumentDiagnosticReport) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 RelatedFullDocumentDiagnosticReport
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 RelatedUnchangedDocumentDiagnosticReport
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [RelatedFullDocumentDiagnosticReport RelatedUnchangedDocumentDiagnosticReport]"}
}

// from line 3895
func (t Or_DocumentDiagnosticReportPartialResult_relatedDocuments_Value) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case FullDocumentDiagnosticReport:
		return json.Marshal(x)
	case UnchangedDocumentDiagnosticReport:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [FullDocumentDiagnosticReport UnchangedDocumentDiagnosticReport]", t)
}

func (t *Or_DocumentDiagnosticReportPartialResult_relatedDocuments_Value) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 FullDocumentDiagnosticReport
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 UnchangedDocumentDiagnosticReport
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [FullDocumentDiagnosticReport UnchangedDocumentDiagnosticReport]"}
}

// from line 14510
func (t Or_DocumentFilter) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case NotebookCellTextDocumentFilter:
		return json.Marshal(x)
	case TextDocumentFilter:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [NotebookCellTextDocumentFilter TextDocumentFilter]", t)
}

func (t *Or_DocumentFilter) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 NotebookCellTextDocumentFilter
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 TextDocumentFilter
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [NotebookCellTextDocumentFilter TextDocumentFilter]"}
}

// from line 5086
func (t Or_Hover_contents) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case MarkedString:
		return json.Marshal(x)
	case MarkupContent:
		return json.Marshal(x)
	case []MarkedString:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [MarkedString MarkupContent []MarkedString]", t)
}

func (t *Or_Hover_contents) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 MarkedString
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 MarkupContent
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	var h2 []MarkedString
	if err := json.Unmarshal(x, &h2); err == nil {
		t.Value = h2
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [MarkedString MarkupContent []MarkedString]"}
}

// from line 3731
func (t Or_InlayHint_label) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case []InlayHintLabelPart:
		return json.Marshal(x)
	case string:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [[]InlayHintLabelPart string]", t)
}

func (t *Or_InlayHint_label) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 []InlayHintLabelPart
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 string
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [[]InlayHintLabelPart string]"}
}

// from line 4163
func (t Or_InlineCompletionItem_insertText) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case StringValue:
		return json.Marshal(x)
	case string:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [StringValue string]", t)
}

func (t *Or_InlineCompletionItem_insertText) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 StringValue
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 string
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [StringValue string]"}
}

// from line 14278
func (t Or_InlineValue) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case InlineValueEvaluatableExpression:
		return json.Marshal(x)
	case InlineValueText:
		return json.Marshal(x)
	case InlineValueVariableLookup:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [InlineValueEvaluatableExpression InlineValueText InlineValueVariableLookup]", t)
}

func (t *Or_InlineValue) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 InlineValueEvaluatableExpression
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 InlineValueText
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	var h2 InlineValueVariableLookup
	if err := json.Unmarshal(x, &h2); err == nil {
		t.Value = h2
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [InlineValueEvaluatableExpression InlineValueText InlineValueVariableLookup]"}
}

// from line 14475
func (t Or_MarkedString) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case Msg_MarkedString:
		return json.Marshal(x)
	case string:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [Msg_MarkedString string]", t)
}

func (t *Or_MarkedString) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 Msg_MarkedString
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 string
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [Msg_MarkedString string]"}
}

// from line 10472
func (t Or_NotebookCellTextDocumentFilter_notebook) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case NotebookDocumentFilter:
		return json.Marshal(x)
	case string:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [NotebookDocumentFilter string]", t)
}

func (t *Or_NotebookCellTextDocumentFilter_notebook) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 NotebookDocumentFilter
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 string
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [NotebookDocumentFilter string]"}
}

// from line 10211
func (t Or_NotebookDocumentSyncOptions_notebookSelector_Elem_Item1_notebook) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case NotebookDocumentFilter:
		return json.Marshal(x)
	case string:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [NotebookDocumentFilter string]", t)
}

func (t *Or_NotebookDocumentSyncOptions_notebookSelector_Elem_Item1_notebook) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 NotebookDocumentFilter
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 string
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [NotebookDocumentFilter string]"}
}

// from line 7404
func (t Or_RelatedFullDocumentDiagnosticReport_relatedDocuments_Value) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case FullDocumentDiagnosticReport:
		return json.Marshal(x)
	case UnchangedDocumentDiagnosticReport:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [FullDocumentDiagnosticReport UnchangedDocumentDiagnosticReport]", t)
}

func (t *Or_RelatedFullDocumentDiagnosticReport_relatedDocuments_Value) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 FullDocumentDiagnosticReport
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 UnchangedDocumentDiagnosticReport
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [FullDocumentDiagnosticReport UnchangedDocumentDiagnosticReport]"}
}

// from line 7443
func (t Or_RelatedUnchangedDocumentDiagnosticReport_relatedDocuments_Value) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case FullDocumentDiagnosticReport:
		return json.Marshal(x)
	case UnchangedDocumentDiagnosticReport:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [FullDocumentDiagnosticReport UnchangedDocumentDiagnosticReport]", t)
}

func (t *Or_RelatedUnchangedDocumentDiagnosticReport_relatedDocuments_Value) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 FullDocumentDiagnosticReport
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 UnchangedDocumentDiagnosticReport
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [FullDocumentDiagnosticReport UnchangedDocumentDiagnosticReport]"}
}

// from line 11106
func (t Or_RelativePattern_baseUri) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case URI:
		return json.Marshal(x)
	case WorkspaceFolder:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [URI WorkspaceFolder]", t)
}

func (t *Or_RelativePattern_baseUri) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 URI
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 WorkspaceFolder
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [URI WorkspaceFolder]"}
}

// from line 1413
func (t Or_Result_textDocument_codeAction_Item0_Elem) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case CodeAction:
		return json.Marshal(x)
	case Command:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [CodeAction Command]", t)
}

func (t *Or_Result_textDocument_codeAction_Item0_Elem) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 CodeAction
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 Command
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [CodeAction Command]"}
}

// from line 980
func (t Or_Result_textDocument_inlineCompletion) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case InlineCompletionList:
		return json.Marshal(x)
	case []InlineCompletionItem:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [InlineCompletionList []InlineCompletionItem]", t)
}

func (t *Or_Result_textDocument_inlineCompletion) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 InlineCompletionList
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 []InlineCompletionItem
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [InlineCompletionList []InlineCompletionItem]"}
}

// from line 12573
func (t Or_SemanticTokensClientCapabilities_requests_full) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case FFullPRequests:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [FFullPRequests bool]", t)
}

func (t *Or_SemanticTokensClientCapabilities_requests_full) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 FFullPRequests
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 bool
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [FFullPRequests bool]"}
}

// from line 12553
func (t Or_SemanticTokensClientCapabilities_requests_range) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case FRangePRequests:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [FRangePRequests bool]", t)
}

func (t *Or_SemanticTokensClientCapabilities_requests_range) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 FRangePRequests
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 bool
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [FRangePRequests bool]"}
}

// from line 6815
func (t Or_SemanticTokensOptions_full) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case PFullESemanticTokensOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [PFullESemanticTokensOptions bool]", t)
}

func (t *Or_SemanticTokensOptions_full) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 PFullESemanticTokensOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 bool
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [PFullESemanticTokensOptions bool]"}
}

// from line 6795
func (t Or_SemanticTokensOptions_range) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case PRangeESemanticTokensOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [PRangeESemanticTokensOptions bool]", t)
}

func (t *Or_SemanticTokensOptions_range) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 PRangeESemanticTokensOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 bool
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [PRangeESemanticTokensOptions bool]"}
}

// from line 8525
func (t Or_ServerCapabilities_callHierarchyProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case CallHierarchyOptions:
		return json.Marshal(x)
	case CallHierarchyRegistrationOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [CallHierarchyOptions CallHierarchyRegistrationOptions bool]", t)
}

func (t *Or_ServerCapabilities_callHierarchyProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 CallHierarchyOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 CallHierarchyRegistrationOptions
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	var h2 bool
	if err := json.Unmarshal(x, &h2); err == nil {
		t.Value = h2
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [CallHierarchyOptions CallHierarchyRegistrationOptions bool]"}
}

// from line 8333
func (t Or_ServerCapabilities_codeActionProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case CodeActionOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [CodeActionOptions bool]", t)
}

func (t *Or_ServerCapabilities_codeActionProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 CodeActionOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 bool
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [CodeActionOptions bool]"}
}

// from line 8369
func (t Or_ServerCapabilities_colorProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case DocumentColorOptions:
		return json.Marshal(x)
	case DocumentColorRegistrationOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [DocumentColorOptions DocumentColorRegistrationOptions bool]", t)
}

func (t *Or_ServerCapabilities_colorProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 DocumentColorOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 DocumentColorRegistrationOptions
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	var h2 bool
	if err := json.Unmarshal(x, &h2); err == nil {
		t.Value = h2
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [DocumentColorOptions DocumentColorRegistrationOptions bool]"}
}

// from line 8195
func (t Or_ServerCapabilities_declarationProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case DeclarationOptions:
		return json.Marshal(x)
	case DeclarationRegistrationOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [DeclarationOptions DeclarationRegistrationOptions bool]", t)
}

func (t *Or_ServerCapabilities_declarationProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 DeclarationOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 DeclarationRegistrationOptions
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	var h2 bool
	if err := json.Unmarshal(x, &h2); err == nil {
		t.Value = h2
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [DeclarationOptions DeclarationRegistrationOptions bool]"}
}

// from line 8217
func (t Or_ServerCapabilities_definitionProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case DefinitionOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [DefinitionOptions bool]", t)
}

func (t *Or_ServerCapabilities_definitionProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 DefinitionOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 bool
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [DefinitionOptions bool]"}
}

// from line 8682
func (t Or_ServerCapabilities_diagnosticProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case DiagnosticOptions:
		return json.Marshal(x)
	case DiagnosticRegistrationOptions:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [DiagnosticOptions DiagnosticRegistrationOptions]", t)
}

func (t *Or_ServerCapabilities_diagnosticProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 DiagnosticOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 DiagnosticRegistrationOptions
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [DiagnosticOptions DiagnosticRegistrationOptions]"}
}

// from line 8409
func (t Or_ServerCapabilities_documentFormattingProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case DocumentFormattingOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [DocumentFormattingOptions bool]", t)
}

func (t *Or_ServerCapabilities_documentFormattingProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 DocumentFormattingOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 bool
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [DocumentFormattingOptions bool]"}
}

// from line 8297
func (t Or_ServerCapabilities_documentHighlightProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case DocumentHighlightOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [DocumentHighlightOptions bool]", t)
}

func (t *Or_ServerCapabilities_documentHighlightProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 DocumentHighlightOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 bool
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [DocumentHighlightOptions bool]"}
}

// from line 8427
func (t Or_ServerCapabilities_documentRangeFormattingProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case DocumentRangeFormattingOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [DocumentRangeFormattingOptions bool]", t)
}

func (t *Or_ServerCapabilities_documentRangeFormattingProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 DocumentRangeFormattingOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 bool
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [DocumentRangeFormattingOptions bool]"}
}

// from line 8315
func (t Or_ServerCapabilities_documentSymbolProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case DocumentSymbolOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [DocumentSymbolOptions bool]", t)
}

func (t *Or_ServerCapabilities_documentSymbolProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 DocumentSymbolOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 bool
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [DocumentSymbolOptions bool]"}
}

// from line 8472
func (t Or_ServerCapabilities_foldingRangeProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case FoldingRangeOptions:
		return json.Marshal(x)
	case FoldingRangeRegistrationOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [FoldingRangeOptions FoldingRangeRegistrationOptions bool]", t)
}

func (t *Or_ServerCapabilities_foldingRangeProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 FoldingRangeOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 FoldingRangeRegistrationOptions
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	var h2 bool
	if err := json.Unmarshal(x, &h2); err == nil {
		t.Value = h2
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [FoldingRangeOptions FoldingRangeRegistrationOptions bool]"}
}

// from line 8168
func (t Or_ServerCapabilities_hoverProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case HoverOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [HoverOptions bool]", t)
}

func (t *Or_ServerCapabilities_hoverProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 HoverOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 bool
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [HoverOptions bool]"}
}

// from line 8257
func (t Or_ServerCapabilities_implementationProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case ImplementationOptions:
		return json.Marshal(x)
	case ImplementationRegistrationOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [ImplementationOptions ImplementationRegistrationOptions bool]", t)
}

func (t *Or_ServerCapabilities_implementationProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 ImplementationOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 ImplementationRegistrationOptions
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	var h2 bool
	if err := json.Unmarshal(x, &h2); err == nil {
		t.Value = h2
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [ImplementationOptions ImplementationRegistrationOptions bool]"}
}

// from line 8659
func (t Or_ServerCapabilities_inlayHintProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case InlayHintOptions:
		return json.Marshal(x)
	case InlayHintRegistrationOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [InlayHintOptions InlayHintRegistrationOptions bool]", t)
}

func (t *Or_ServerCapabilities_inlayHintProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 InlayHintOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 InlayHintRegistrationOptions
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	var h2 bool
	if err := json.Unmarshal(x, &h2); err == nil {
		t.Value = h2
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [InlayHintOptions InlayHintRegistrationOptions bool]"}
}

// from line 8701
func (t Or_ServerCapabilities_inlineCompletionProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case InlineCompletionOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [InlineCompletionOptions bool]", t)
}

func (t *Or_ServerCapabilities_inlineCompletionProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 InlineCompletionOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 bool
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [InlineCompletionOptions bool]"}
}

// from line 8636
func (t Or_ServerCapabilities_inlineValueProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case InlineValueOptions:
		return json.Marshal(x)
	case InlineValueRegistrationOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [InlineValueOptions InlineValueRegistrationOptions bool]", t)
}

func (t *Or_ServerCapabilities_inlineValueProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 InlineValueOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 InlineValueRegistrationOptions
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	var h2 bool
	if err := json.Unmarshal(x, &h2); err == nil {
		t.Value = h2
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [InlineValueOptions InlineValueRegistrationOptions bool]"}
}

// from line 8548
func (t Or_ServerCapabilities_linkedEditingRangeProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case LinkedEditingRangeOptions:
		return json.Marshal(x)
	case LinkedEditingRangeRegistrationOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [LinkedEditingRangeOptions LinkedEditingRangeRegistrationOptions bool]", t)
}

func (t *Or_ServerCapabilities_linkedEditingRangeProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 LinkedEditingRangeOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 LinkedEditingRangeRegistrationOptions
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	var h2 bool
	if err := json.Unmarshal(x, &h2); err == nil {
		t.Value = h2
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [LinkedEditingRangeOptions LinkedEditingRangeRegistrationOptions bool]"}
}

// from line 8590
func (t Or_ServerCapabilities_monikerProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case MonikerOptions:
		return json.Marshal(x)
	case MonikerRegistrationOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [MonikerOptions MonikerRegistrationOptions bool]", t)
}

func (t *Or_ServerCapabilities_monikerProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 MonikerOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 MonikerRegistrationOptions
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	var h2 bool
	if err := json.Unmarshal(x, &h2); err == nil {
		t.Value = h2
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [MonikerOptions MonikerRegistrationOptions bool]"}
}

// from line 8140
func (t Or_ServerCapabilities_notebookDocumentSync) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case NotebookDocumentSyncOptions:
		return json.Marshal(x)
	case NotebookDocumentSyncRegistrationOptions:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [NotebookDocumentSyncOptions NotebookDocumentSyncRegistrationOptions]", t)
}

func (t *Or_ServerCapabilities_notebookDocumentSync) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 NotebookDocumentSyncOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 NotebookDocumentSyncRegistrationOptions
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [NotebookDocumentSyncOptions NotebookDocumentSyncRegistrationOptions]"}
}

// from line 8279
func (t Or_ServerCapabilities_referencesProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case ReferenceOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [ReferenceOptions bool]", t)
}

func (t *Or_ServerCapabilities_referencesProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 ReferenceOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 bool
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [ReferenceOptions bool]"}
}

// from line 8454
func (t Or_ServerCapabilities_renameProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case RenameOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [RenameOptions bool]", t)
}

func (t *Or_ServerCapabilities_renameProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 RenameOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 bool
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [RenameOptions bool]"}
}

// from line 8494
func (t Or_ServerCapabilities_selectionRangeProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case SelectionRangeOptions:
		return json.Marshal(x)
	case SelectionRangeRegistrationOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [SelectionRangeOptions SelectionRangeRegistrationOptions bool]", t)
}

func (t *Or_ServerCapabilities_selectionRangeProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 SelectionRangeOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 SelectionRangeRegistrationOptions
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	var h2 bool
	if err := json.Unmarshal(x, &h2); err == nil {
		t.Value = h2
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [SelectionRangeOptions SelectionRangeRegistrationOptions bool]"}
}

// from line 8571
func (t Or_ServerCapabilities_semanticTokensProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case SemanticTokensOptions:
		return json.Marshal(x)
	case SemanticTokensRegistrationOptions:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [SemanticTokensOptions SemanticTokensRegistrationOptions]", t)
}

func (t *Or_ServerCapabilities_semanticTokensProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 SemanticTokensOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 SemanticTokensRegistrationOptions
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [SemanticTokensOptions SemanticTokensRegistrationOptions]"}
}

// from line 8122
func (t Or_ServerCapabilities_textDocumentSync) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case TextDocumentSyncKind:
		return json.Marshal(x)
	case TextDocumentSyncOptions:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [TextDocumentSyncKind TextDocumentSyncOptions]", t)
}

func (t *Or_ServerCapabilities_textDocumentSync) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 TextDocumentSyncKind
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 TextDocumentSyncOptions
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [TextDocumentSyncKind TextDocumentSyncOptions]"}
}

// from line 8235
func (t Or_ServerCapabilities_typeDefinitionProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case TypeDefinitionOptions:
		return json.Marshal(x)
	case TypeDefinitionRegistrationOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [TypeDefinitionOptions TypeDefinitionRegistrationOptions bool]", t)
}

func (t *Or_ServerCapabilities_typeDefinitionProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 TypeDefinitionOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 TypeDefinitionRegistrationOptions
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	var h2 bool
	if err := json.Unmarshal(x, &h2); err == nil {
		t.Value = h2
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [TypeDefinitionOptions TypeDefinitionRegistrationOptions bool]"}
}

// from line 8613
func (t Or_ServerCapabilities_typeHierarchyProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case TypeHierarchyOptions:
		return json.Marshal(x)
	case TypeHierarchyRegistrationOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [TypeHierarchyOptions TypeHierarchyRegistrationOptions bool]", t)
}

func (t *Or_ServerCapabilities_typeHierarchyProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 TypeHierarchyOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 TypeHierarchyRegistrationOptions
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	var h2 bool
	if err := json.Unmarshal(x, &h2); err == nil {
		t.Value = h2
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [TypeHierarchyOptions TypeHierarchyRegistrationOptions bool]"}
}

// from line 8391
func (t Or_ServerCapabilities_workspaceSymbolProvider) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case WorkspaceSymbolOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [WorkspaceSymbolOptions bool]", t)
}

func (t *Or_ServerCapabilities_workspaceSymbolProvider) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 WorkspaceSymbolOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 bool
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [WorkspaceSymbolOptions bool]"}
}

// from line 9159
func (t Or_SignatureInformation_documentation) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case MarkupContent:
		return json.Marshal(x)
	case string:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [MarkupContent string]", t)
}

func (t *Or_SignatureInformation_documentation) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 MarkupContent
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 string
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [MarkupContent string]"}
}

// from line 6928
func (t Or_TextDocumentEdit_edits_Elem) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case AnnotatedTextEdit:
		return json.Marshal(x)
	case TextEdit:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [AnnotatedTextEdit TextEdit]", t)
}

func (t *Or_TextDocumentEdit_edits_Elem) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 AnnotatedTextEdit
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 TextEdit
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [AnnotatedTextEdit TextEdit]"}
}

// from line 10131
func (t Or_TextDocumentSyncOptions_save) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case SaveOptions:
		return json.Marshal(x)
	case bool:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [SaveOptions bool]", t)
}

func (t *Or_TextDocumentSyncOptions_save) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 SaveOptions
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 bool
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [SaveOptions bool]"}
}

// from line 14401
func (t Or_WorkspaceDocumentDiagnosticReport) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case WorkspaceFullDocumentDiagnosticReport:
		return json.Marshal(x)
	case WorkspaceUnchangedDocumentDiagnosticReport:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [WorkspaceFullDocumentDiagnosticReport WorkspaceUnchangedDocumentDiagnosticReport]", t)
}

func (t *Or_WorkspaceDocumentDiagnosticReport) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 WorkspaceFullDocumentDiagnosticReport
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 WorkspaceUnchangedDocumentDiagnosticReport
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [WorkspaceFullDocumentDiagnosticReport WorkspaceUnchangedDocumentDiagnosticReport]"}
}

// from line 3292
func (t Or_WorkspaceEdit_documentChanges_Elem) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case CreateFile:
		return json.Marshal(x)
	case DeleteFile:
		return json.Marshal(x)
	case RenameFile:
		return json.Marshal(x)
	case TextDocumentEdit:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [CreateFile DeleteFile RenameFile TextDocumentEdit]", t)
}

func (t *Or_WorkspaceEdit_documentChanges_Elem) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 CreateFile
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 DeleteFile
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	var h2 RenameFile
	if err := json.Unmarshal(x, &h2); err == nil {
		t.Value = h2
		return nil
	}
	var h3 TextDocumentEdit
	if err := json.Unmarshal(x, &h3); err == nil {
		t.Value = h3
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [CreateFile DeleteFile RenameFile TextDocumentEdit]"}
}

// from line 248
func (t Or_textDocument_declaration) MarshalJSON() ([]byte, error) {
	switch x := t.Value.(type) {
	case Declaration:
		return json.Marshal(x)
	case []DeclarationLink:
		return json.Marshal(x)
	case nil:
		return []byte("null"), nil
	}
	return nil, fmt.Errorf("type %T not one of [Declaration []DeclarationLink]", t)
}

func (t *Or_textDocument_declaration) UnmarshalJSON(x []byte) error {
	if string(x) == "null" {
		t.Value = nil
		return nil
	}
	var h0 Declaration
	if err := json.Unmarshal(x, &h0); err == nil {
		t.Value = h0
		return nil
	}
	var h1 []DeclarationLink
	if err := json.Unmarshal(x, &h1); err == nil {
		t.Value = h1
		return nil
	}
	return &UnmarshalError{"unmarshal failed to match one of [Declaration []DeclarationLink]"}
}
