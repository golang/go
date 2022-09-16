// Copyright 2019-2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocol

// Code generated from version 3.17.0 of protocol/metaModel.json.
// git hash 8de18faed635819dd2bc631d2c26ce4a18f7cf4a (as of Fri Sep 16 13:04:31 2022)
// Code generated; DO NOT EDIT.

import "encoding/json"
import "errors"
import "fmt"

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
	return errors.New("unmarshal failed to match one of [FEditRangePItemDefaults Range]")
}

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
	return errors.New("unmarshal failed to match one of [NotebookDocumentFilter string]")
}

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
	return errors.New("unmarshal failed to match one of [Location PLocationMsg_workspace_symbol]")
}

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
	return errors.New("unmarshal failed to match one of [[]string string]")
}

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
	return errors.New("unmarshal failed to match one of [MarkupContent string]")
}

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
	return errors.New("unmarshal failed to match one of [MarkupContent string]")
}

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
	return errors.New("unmarshal failed to match one of [Location []Location]")
}

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
	return errors.New("unmarshal failed to match one of [RelatedFullDocumentDiagnosticReport RelatedUnchangedDocumentDiagnosticReport]")
}

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
	return errors.New("unmarshal failed to match one of [NotebookCellTextDocumentFilter TextDocumentFilter]")
}

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
	return errors.New("unmarshal failed to match one of [InlineValueEvaluatableExpression InlineValueText InlineValueVariableLookup]")
}

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
	return errors.New("unmarshal failed to match one of [Msg_MarkedString string]")
}

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
	return errors.New("unmarshal failed to match one of [URI WorkspaceFolder]")
}

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
	return errors.New("unmarshal failed to match one of [WorkspaceFullDocumentDiagnosticReport WorkspaceUnchangedDocumentDiagnosticReport]")
}

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
	return errors.New("unmarshal failed to match one of [Declaration []DeclarationLink]")
}
