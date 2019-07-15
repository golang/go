// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wire

// This file holds common ocagent types

type Node struct {
	Identifier  *ProcessIdentifier `json:"identifier,omitempty"`
	LibraryInfo *LibraryInfo       `json:"library_info,omitempty"`
	ServiceInfo *ServiceInfo       `json:"service_info,omitempty"`
	Attributes  map[string]string  `json:"attributes,omitempty"`
}

type Resource struct {
	Type   string            `json:"type,omitempty"`
	Labels map[string]string `json:"labels,omitempty"`
}

type TruncatableString struct {
	Value              string `json:"value,omitempty"`
	TruncatedByteCount int32  `json:"truncated_byte_count,omitempty"`
}

type Attributes struct {
	AttributeMap           map[string]Attribute `json:"attributeMap,omitempty"`
	DroppedAttributesCount int32                `json:"dropped_attributes_count,omitempty"`
}

type StringAttribute struct {
	StringValue *TruncatableString `json:"stringValue,omitempty"`
}

type IntAttribute struct {
	IntValue int64 `json:"intValue,omitempty"`
}

type BoolAttribute struct {
	BoolValue bool `json:"boolValue,omitempty"`
}

type DoubleAttribute struct {
	DoubleValue float64 `json:"doubleValue,omitempty"`
}

type Attribute interface {
	tagAttribute()
}

func (StringAttribute) tagAttribute() {}
func (IntAttribute) tagAttribute()    {}
func (BoolAttribute) tagAttribute()   {}
func (DoubleAttribute) tagAttribute() {}

type StackTrace struct {
	StackFrames      *StackFrames `json:"stack_frames,omitempty"`
	StackTraceHashId uint64       `json:"stack_trace_hash_id,omitempty"`
}

type StackFrames struct {
	Frame              []*StackFrame `json:"frame,omitempty"`
	DroppedFramesCount int32         `json:"dropped_frames_count,omitempty"`
}

type StackFrame struct {
	FunctionName         *TruncatableString `json:"function_name,omitempty"`
	OriginalFunctionName *TruncatableString `json:"original_function_name,omitempty"`
	FileName             *TruncatableString `json:"file_name,omitempty"`
	LineNumber           int64              `json:"line_number,omitempty"`
	ColumnNumber         int64              `json:"column_number,omitempty"`
	LoadModule           *Module            `json:"load_module,omitempty"`
	SourceVersion        *TruncatableString `json:"source_version,omitempty"`
}

type Module struct {
	Module  *TruncatableString `json:"module,omitempty"`
	BuildId *TruncatableString `json:"build_id,omitempty"`
}

type ProcessIdentifier struct {
	HostName       string    `json:"host_name,omitempty"`
	Pid            uint32    `json:"pid,omitempty"`
	StartTimestamp Timestamp `json:"start_timestamp,omitempty"`
}

type LibraryInfo struct {
	Language           Language `json:"language,omitempty"`
	ExporterVersion    string   `json:"exporter_version,omitempty"`
	CoreLibraryVersion string   `json:"core_library_version,omitempty"`
}

type Language int32

const (
	LanguageGo Language = 4
)

type ServiceInfo struct {
	Name string `json:"name,omitempty"`
}
