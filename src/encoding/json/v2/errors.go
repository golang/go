// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"cmp"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"sync"

	"encoding/json/internal/jsonflags"
	"encoding/json/internal/jsonopts"
	"encoding/json/internal/jsonwire"
	"encoding/json/jsontext"
)

// ErrUnknownName indicates that a JSON object member could not be
// unmarshaled because the name is not known to the target Go struct.
// This error is directly wrapped within a [SemanticError] when produced.
//
// The name of an unknown JSON object member can be extracted as:
//
//	err := ...
//	var serr json.SemanticError
//	if errors.As(err, &serr) && serr.Err == json.ErrUnknownName {
//		ptr := serr.JSONPointer // JSON pointer to unknown name
//		name := ptr.LastToken() // unknown name itself
//		...
//	}
//
// This error is only returned if [RejectUnknownMembers] is true.
var ErrUnknownName = errors.New("unknown object member name")

const errorPrefix = "json: "

func isSemanticError(err error) bool {
	_, ok := err.(*SemanticError)
	return ok
}

func isSyntacticError(err error) bool {
	_, ok := err.(*jsontext.SyntacticError)
	return ok
}

// isFatalError reports whether this error must terminate asharling.
// All errors are considered fatal unless operating under
// [jsonflags.ReportErrorsWithLegacySemantics] in which case only
// syntactic errors and I/O errors are considered fatal.
func isFatalError(err error, flags jsonflags.Flags) bool {
	return !flags.Get(jsonflags.ReportErrorsWithLegacySemantics) ||
		isSyntacticError(err) || export.IsIOError(err)
}

// SemanticError describes an error determining the meaning
// of JSON data as Go data or vice-versa.
//
// If a [Marshaler], [MarshalerTo], [Unmarshaler], or [UnmarshalerFrom] method
// returns a SemanticError when called by the [json] package,
// then the ByteOffset, JSONPointer, and GoType fields are automatically
// populated by the calling context if they are the zero value.
//
// The contents of this error as produced by this package may change over time.
type SemanticError struct {
	requireKeyedLiterals
	nonComparable

	action string // either "marshal" or "unmarshal"

	// ByteOffset indicates that an error occurred after this byte offset.
	ByteOffset int64
	// JSONPointer indicates that an error occurred within this JSON value
	// as indicated using the JSON Pointer notation (see RFC 6901).
	JSONPointer jsontext.Pointer

	// JSONKind is the JSON kind that could not be handled.
	JSONKind jsontext.Kind // may be zero if unknown
	// JSONValue is the JSON number or string that could not be unmarshaled.
	// It is not populated during marshaling.
	JSONValue jsontext.Value // may be nil if irrelevant or unknown
	// GoType is the Go type that could not be handled.
	GoType reflect.Type // may be nil if unknown

	// Err is the underlying error.
	Err error // may be nil
}

// coder is implemented by [jsontext.Encoder] or [jsontext.Decoder].
type coder interface {
	StackPointer() jsontext.Pointer
	Options() Options
}

// newInvalidFormatError wraps err in a SemanticError because
// the current type t cannot handle the provided options format.
// This error must be called before producing or consuming the next value.
//
// If [jsonflags.ReportErrorsWithLegacySemantics] is specified,
// then this automatically skips the next value when unmarshaling
// to ensure that the value is fully consumed.
func newInvalidFormatError(c coder, t reflect.Type) error {
	err := fmt.Errorf("invalid format flag %q", c.Options().(*jsonopts.Struct).Format)
	switch c := c.(type) {
	case *jsontext.Encoder:
		err = newMarshalErrorBefore(c, t, err)
	case *jsontext.Decoder:
		err = newUnmarshalErrorBeforeWithSkipping(c, t, err)
	}
	return err
}

// newMarshalErrorBefore wraps err in a SemanticError assuming that e
// is positioned right before the next token or value, which causes an error.
func newMarshalErrorBefore(e *jsontext.Encoder, t reflect.Type, err error) error {
	return &SemanticError{action: "marshal", GoType: t, Err: err,
		ByteOffset:  e.OutputOffset() + int64(export.Encoder(e).CountNextDelimWhitespace()),
		JSONPointer: jsontext.Pointer(export.Encoder(e).AppendStackPointer(nil, +1))}
}

// newUnmarshalErrorBefore wraps err in a SemanticError assuming that d
// is positioned right before the next token or value, which causes an error.
// It does not record the next JSON kind as this error is used to indicate
// the receiving Go value is invalid to unmarshal into (and not a JSON error).
// However, if [jsonflags.ReportErrorsWithLegacySemantics] is specified,
// then it does record the next JSON kind for historical reporting reasons.
func newUnmarshalErrorBefore(d *jsontext.Decoder, t reflect.Type, err error) error {
	var k jsontext.Kind
	if export.Decoder(d).Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
		k = d.PeekKind()
	}
	return &SemanticError{action: "unmarshal", GoType: t, Err: err,
		ByteOffset:  d.InputOffset() + int64(export.Decoder(d).CountNextDelimWhitespace()),
		JSONPointer: jsontext.Pointer(export.Decoder(d).AppendStackPointer(nil, +1)),
		JSONKind:    k}
}

// newUnmarshalErrorBeforeWithSkipping is like [newUnmarshalErrorBefore],
// but automatically skips the next value if
// [jsonflags.ReportErrorsWithLegacySemantics] is specified.
func newUnmarshalErrorBeforeWithSkipping(d *jsontext.Decoder, t reflect.Type, err error) error {
	err = newUnmarshalErrorBefore(d, t, err)
	if export.Decoder(d).Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
		if err2 := export.Decoder(d).SkipValue(); err2 != nil {
			return err2
		}
	}
	return err
}

// newUnmarshalErrorAfter wraps err in a SemanticError assuming that d
// is positioned right after the previous token or value, which caused an error.
func newUnmarshalErrorAfter(d *jsontext.Decoder, t reflect.Type, err error) error {
	tokOrVal := export.Decoder(d).PreviousTokenOrValue()
	return &SemanticError{action: "unmarshal", GoType: t, Err: err,
		ByteOffset:  d.InputOffset() - int64(len(tokOrVal)),
		JSONPointer: jsontext.Pointer(export.Decoder(d).AppendStackPointer(nil, -1)),
		JSONKind:    jsontext.Value(tokOrVal).Kind()}
}

// newUnmarshalErrorAfter wraps err in a SemanticError assuming that d
// is positioned right after the previous token or value, which caused an error.
// It also stores a copy of the last JSON value if it is a string or number.
func newUnmarshalErrorAfterWithValue(d *jsontext.Decoder, t reflect.Type, err error) error {
	serr := newUnmarshalErrorAfter(d, t, err).(*SemanticError)
	if serr.JSONKind == '"' || serr.JSONKind == '0' {
		serr.JSONValue = jsontext.Value(export.Decoder(d).PreviousTokenOrValue()).Clone()
	}
	return serr
}

// newUnmarshalErrorAfterWithSkipping is like [newUnmarshalErrorAfter],
// but automatically skips the remainder of the current value if
// [jsonflags.ReportErrorsWithLegacySemantics] is specified.
func newUnmarshalErrorAfterWithSkipping(d *jsontext.Decoder, t reflect.Type, err error) error {
	err = newUnmarshalErrorAfter(d, t, err)
	if export.Decoder(d).Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
		if err2 := export.Decoder(d).SkipValueRemainder(); err2 != nil {
			return err2
		}
	}
	return err
}

// newSemanticErrorWithPosition wraps err in a SemanticError assuming that
// the error occurred at the provided depth, and length.
// If err is already a SemanticError, then position information is only
// injected if it is currently unpopulated.
//
// If the position is unpopulated, it is ambiguous where the error occurred
// in the user code, whether it was before or after the current position.
// For the byte offset, we assume that the error occurred before the last read
// token or value when decoding, or before the next value when encoding.
// For the JSON pointer, we point to the parent object or array unless
// we can be certain that it happened with an object member.
//
// This is used to annotate errors returned by user-provided
// v2 MarshalJSON or UnmarshalJSON methods or functions.
func newSemanticErrorWithPosition(c coder, t reflect.Type, prevDepth int, prevLength int64, err error) error {
	serr, _ := err.(*SemanticError)
	if serr == nil {
		serr = &SemanticError{Err: err}
	}
	var currDepth int
	var currLength int64
	var coderState interface{ AppendStackPointer([]byte, int) []byte }
	var offset int64
	switch c := c.(type) {
	case *jsontext.Encoder:
		e := export.Encoder(c)
		serr.action = cmp.Or(serr.action, "marshal")
		currDepth, currLength = e.Tokens.DepthLength()
		offset = c.OutputOffset() + int64(export.Encoder(c).CountNextDelimWhitespace())
		coderState = e
	case *jsontext.Decoder:
		d := export.Decoder(c)
		serr.action = cmp.Or(serr.action, "unmarshal")
		currDepth, currLength = d.Tokens.DepthLength()
		tokOrVal := d.PreviousTokenOrValue()
		offset = c.InputOffset() - int64(len(tokOrVal))
		if (prevDepth == currDepth && prevLength == currLength) || len(tokOrVal) == 0 {
			// If no Read method was called in the user-defined method or
			// if the Peek method was called, then use the offset of the next value.
			offset = c.InputOffset() + int64(export.Decoder(c).CountNextDelimWhitespace())
		}
		coderState = d
	}
	serr.ByteOffset = cmp.Or(serr.ByteOffset, offset)
	if serr.JSONPointer == "" {
		where := 0 // default to ambiguous positioning
		switch {
		case prevDepth == currDepth && prevLength+0 == currLength:
			where = +1
		case prevDepth == currDepth && prevLength+1 == currLength:
			where = -1
		}
		serr.JSONPointer = jsontext.Pointer(coderState.AppendStackPointer(nil, where))
	}
	serr.GoType = cmp.Or(serr.GoType, t)
	return serr
}

// collapseSemanticErrors collapses double SemanticErrors at the outer levels
// into a single SemanticError by preserving the inner error,
// but prepending the ByteOffset and JSONPointer with the outer error.
//
// For example:
//
//	collapseSemanticErrors(&SemanticError{
//		ByteOffset:  len64(`[0,{"alpha":[0,1,`),
//		JSONPointer: "/1/alpha/2",
//		GoType:      reflect.TypeFor[outerType](),
//		Err: &SemanticError{
//			ByteOffset:  len64(`{"foo":"bar","fizz":[0,`),
//			JSONPointer: "/fizz/1",
//			GoType:      reflect.TypeFor[innerType](),
//			Err:         ...,
//		},
//	})
//
// results in:
//
//	&SemanticError{
//		ByteOffset:  len64(`[0,{"alpha":[0,1,`) + len64(`{"foo":"bar","fizz":[0,`),
//		JSONPointer: "/1/alpha/2" + "/fizz/1",
//		GoType:      reflect.TypeFor[innerType](),
//		Err:         ...,
//	}
//
// This is used to annotate errors returned by user-provided
// v1 MarshalJSON or UnmarshalJSON methods with precise position information
// if they themselves happened to return a SemanticError.
// Since MarshalJSON and UnmarshalJSON are not operating on the root JSON value,
// their positioning must be relative to the nested JSON value
// returned by UnmarshalJSON or passed to MarshalJSON.
// Therefore, we can construct an absolute position by concatenating
// the outer with the inner positions.
//
// Note that we do not use collapseSemanticErrors with user-provided functions
// that take in an [jsontext.Encoder] or [jsontext.Decoder] since they contain
// methods to report position relative to the root JSON value.
// We assume user-constructed errors are correctly precise about position.
func collapseSemanticErrors(err error) error {
	if serr1, ok := err.(*SemanticError); ok {
		if serr2, ok := serr1.Err.(*SemanticError); ok {
			serr2.ByteOffset = serr1.ByteOffset + serr2.ByteOffset
			serr2.JSONPointer = serr1.JSONPointer + serr2.JSONPointer
			*serr1 = *serr2
		}
	}
	return err
}

// errorModalVerb is a modal verb like "cannot" or "unable to".
//
// Once per process, Hyrum-proof the error message by deliberately
// switching between equivalent renderings of the same error message.
// The randomization is tied to the Hyrum-proofing already applied
// on map iteration in Go.
var errorModalVerb = sync.OnceValue(func() string {
	for phrase := range map[string]struct{}{"cannot": {}, "unable to": {}} {
		return phrase // use whichever phrase we get in the first iteration
	}
	return ""
})

func (e *SemanticError) Error() string {
	var sb strings.Builder
	sb.WriteString(errorPrefix)
	sb.WriteString(errorModalVerb())

	// Format action.
	var preposition string
	switch e.action {
	case "marshal":
		sb.WriteString(" marshal")
		preposition = " from"
	case "unmarshal":
		sb.WriteString(" unmarshal")
		preposition = " into"
	default:
		sb.WriteString(" handle")
		preposition = " with"
	}

	// Format JSON kind.
	switch e.JSONKind {
	case 'n':
		sb.WriteString(" JSON null")
	case 'f', 't':
		sb.WriteString(" JSON boolean")
	case '"':
		sb.WriteString(" JSON string")
	case '0':
		sb.WriteString(" JSON number")
	case '{', '}':
		sb.WriteString(" JSON object")
	case '[', ']':
		sb.WriteString(" JSON array")
	default:
		if e.action == "" {
			preposition = ""
		}
	}
	if len(e.JSONValue) > 0 && len(e.JSONValue) < 100 {
		sb.WriteByte(' ')
		sb.Write(e.JSONValue)
	}

	// Format Go type.
	if e.GoType != nil {
		typeString := e.GoType.String()
		if len(typeString) > 100 {
			// An excessively long type string most likely occurs for
			// an anonymous struct declaration with many fields.
			// Reduce the noise by just printing the kind,
			// and optionally prepending it with the package name
			// if the struct happens to include an unexported field.
			typeString = e.GoType.Kind().String()
			if e.GoType.Kind() == reflect.Struct && e.GoType.Name() == "" {
				for i := range e.GoType.NumField() {
					if pkgPath := e.GoType.Field(i).PkgPath; pkgPath != "" {
						typeString = pkgPath[strings.LastIndexByte(pkgPath, '/')+len("/"):] + ".struct"
						break
					}
				}
			}
		}
		sb.WriteString(preposition)
		sb.WriteString(" Go ")
		sb.WriteString(typeString)
	}

	// Special handling for unknown names.
	if e.Err == ErrUnknownName {
		sb.WriteString(": ")
		sb.WriteString(ErrUnknownName.Error())
		sb.WriteString(" ")
		sb.WriteString(strconv.Quote(e.JSONPointer.LastToken()))
		if parent := e.JSONPointer.Parent(); parent != "" {
			sb.WriteString(" within ")
			sb.WriteString(strconv.Quote(jsonwire.TruncatePointer(string(parent), 100)))
		}
		return sb.String()
	}

	// Format where.
	// Avoid printing if it overlaps with a wrapped SyntacticError.
	switch serr, _ := e.Err.(*jsontext.SyntacticError); {
	case e.JSONPointer != "":
		if serr == nil || !e.JSONPointer.Contains(serr.JSONPointer) {
			sb.WriteString(" within ")
			sb.WriteString(strconv.Quote(jsonwire.TruncatePointer(string(e.JSONPointer), 100)))
		}
	case e.ByteOffset > 0:
		if serr == nil || !(e.ByteOffset <= serr.ByteOffset) {
			sb.WriteString(" after offset ")
			sb.WriteString(strconv.FormatInt(e.ByteOffset, 10))
		}
	}

	// Format underlying error.
	if e.Err != nil {
		errString := e.Err.Error()
		if isSyntacticError(e.Err) {
			errString = strings.TrimPrefix(errString, "jsontext: ")
		}
		sb.WriteString(": ")
		sb.WriteString(errString)
	}

	return sb.String()
}

func (e *SemanticError) Unwrap() error {
	return e.Err
}

func newDuplicateNameError(ptr jsontext.Pointer, quotedName []byte, offset int64) error {
	if quotedName != nil {
		name, _ := jsonwire.AppendUnquote(nil, quotedName)
		ptr = ptr.AppendToken(string(name))
	}
	return &jsontext.SyntacticError{
		ByteOffset:  offset,
		JSONPointer: ptr,
		Err:         jsontext.ErrDuplicateName,
	}
}
