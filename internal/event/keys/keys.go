// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package keys

import (
	"fmt"
	"io"
	"math"
	"strconv"

	"golang.org/x/tools/internal/event/label"
)

// Value represents a key for untyped values.
type Value struct {
	name        string
	description string
}

// New creates a new Key for untyped values.
func New(name, description string) *Value {
	return &Value{name: name, description: description}
}

func (k *Value) Name() string        { return k.name }
func (k *Value) Description() string { return k.description }

func (k *Value) Format(w io.Writer, buf []byte, l label.Label) {
	fmt.Fprint(w, k.From(l))
}

// Get can be used to get a label for the key from a label.Map.
func (k *Value) Get(lm label.Map) interface{} {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return nil
}

// From can be used to get a value from a Label.
func (k *Value) From(t label.Label) interface{} { return t.UnpackValue() }

// Of creates a new Label with this key and the supplied value.
func (k *Value) Of(value interface{}) label.Label { return label.OfValue(k, value) }

// Tag represents a key for tagging labels that have no value.
// These are used when the existence of the label is the entire information it
// carries, such as marking events to be of a specific kind, or from a specific
// package.
type Tag struct {
	name        string
	description string
}

// NewTag creates a new Key for tagging labels.
func NewTag(name, description string) *Tag {
	return &Tag{name: name, description: description}
}

func (k *Tag) Name() string        { return k.name }
func (k *Tag) Description() string { return k.description }

func (k *Tag) Format(w io.Writer, buf []byte, l label.Label) {}

// New creates a new Label with this key.
func (k *Tag) New() label.Label { return label.OfValue(k, nil) }

// Int represents a key
type Int struct {
	name        string
	description string
}

// NewInt creates a new Key for int values.
func NewInt(name, description string) *Int {
	return &Int{name: name, description: description}
}

func (k *Int) Name() string        { return k.name }
func (k *Int) Description() string { return k.description }

func (k *Int) Format(w io.Writer, buf []byte, l label.Label) {
	w.Write(strconv.AppendInt(buf, int64(k.From(l)), 10))
}

// Of creates a new Label with this key and the supplied value.
func (k *Int) Of(v int) label.Label { return label.Of64(k, uint64(v)) }

// Get can be used to get a label for the key from a label.Map.
func (k *Int) Get(lm label.Map) int {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return 0
}

// From can be used to get a value from a Label.
func (k *Int) From(t label.Label) int { return int(t.Unpack64()) }

// Int8 represents a key
type Int8 struct {
	name        string
	description string
}

// NewInt8 creates a new Key for int8 values.
func NewInt8(name, description string) *Int8 {
	return &Int8{name: name, description: description}
}

func (k *Int8) Name() string        { return k.name }
func (k *Int8) Description() string { return k.description }

func (k *Int8) Format(w io.Writer, buf []byte, l label.Label) {
	w.Write(strconv.AppendInt(buf, int64(k.From(l)), 10))
}

// Of creates a new Label with this key and the supplied value.
func (k *Int8) Of(v int8) label.Label { return label.Of64(k, uint64(v)) }

// Get can be used to get a label for the key from a label.Map.
func (k *Int8) Get(lm label.Map) int8 {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return 0
}

// From can be used to get a value from a Label.
func (k *Int8) From(t label.Label) int8 { return int8(t.Unpack64()) }

// Int16 represents a key
type Int16 struct {
	name        string
	description string
}

// NewInt16 creates a new Key for int16 values.
func NewInt16(name, description string) *Int16 {
	return &Int16{name: name, description: description}
}

func (k *Int16) Name() string        { return k.name }
func (k *Int16) Description() string { return k.description }

func (k *Int16) Format(w io.Writer, buf []byte, l label.Label) {
	w.Write(strconv.AppendInt(buf, int64(k.From(l)), 10))
}

// Of creates a new Label with this key and the supplied value.
func (k *Int16) Of(v int16) label.Label { return label.Of64(k, uint64(v)) }

// Get can be used to get a label for the key from a label.Map.
func (k *Int16) Get(lm label.Map) int16 {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return 0
}

// From can be used to get a value from a Label.
func (k *Int16) From(t label.Label) int16 { return int16(t.Unpack64()) }

// Int32 represents a key
type Int32 struct {
	name        string
	description string
}

// NewInt32 creates a new Key for int32 values.
func NewInt32(name, description string) *Int32 {
	return &Int32{name: name, description: description}
}

func (k *Int32) Name() string        { return k.name }
func (k *Int32) Description() string { return k.description }

func (k *Int32) Format(w io.Writer, buf []byte, l label.Label) {
	w.Write(strconv.AppendInt(buf, int64(k.From(l)), 10))
}

// Of creates a new Label with this key and the supplied value.
func (k *Int32) Of(v int32) label.Label { return label.Of64(k, uint64(v)) }

// Get can be used to get a label for the key from a label.Map.
func (k *Int32) Get(lm label.Map) int32 {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return 0
}

// From can be used to get a value from a Label.
func (k *Int32) From(t label.Label) int32 { return int32(t.Unpack64()) }

// Int64 represents a key
type Int64 struct {
	name        string
	description string
}

// NewInt64 creates a new Key for int64 values.
func NewInt64(name, description string) *Int64 {
	return &Int64{name: name, description: description}
}

func (k *Int64) Name() string        { return k.name }
func (k *Int64) Description() string { return k.description }

func (k *Int64) Format(w io.Writer, buf []byte, l label.Label) {
	w.Write(strconv.AppendInt(buf, k.From(l), 10))
}

// Of creates a new Label with this key and the supplied value.
func (k *Int64) Of(v int64) label.Label { return label.Of64(k, uint64(v)) }

// Get can be used to get a label for the key from a label.Map.
func (k *Int64) Get(lm label.Map) int64 {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return 0
}

// From can be used to get a value from a Label.
func (k *Int64) From(t label.Label) int64 { return int64(t.Unpack64()) }

// UInt represents a key
type UInt struct {
	name        string
	description string
}

// NewUInt creates a new Key for uint values.
func NewUInt(name, description string) *UInt {
	return &UInt{name: name, description: description}
}

func (k *UInt) Name() string        { return k.name }
func (k *UInt) Description() string { return k.description }

func (k *UInt) Format(w io.Writer, buf []byte, l label.Label) {
	w.Write(strconv.AppendUint(buf, uint64(k.From(l)), 10))
}

// Of creates a new Label with this key and the supplied value.
func (k *UInt) Of(v uint) label.Label { return label.Of64(k, uint64(v)) }

// Get can be used to get a label for the key from a label.Map.
func (k *UInt) Get(lm label.Map) uint {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return 0
}

// From can be used to get a value from a Label.
func (k *UInt) From(t label.Label) uint { return uint(t.Unpack64()) }

// UInt8 represents a key
type UInt8 struct {
	name        string
	description string
}

// NewUInt8 creates a new Key for uint8 values.
func NewUInt8(name, description string) *UInt8 {
	return &UInt8{name: name, description: description}
}

func (k *UInt8) Name() string        { return k.name }
func (k *UInt8) Description() string { return k.description }

func (k *UInt8) Format(w io.Writer, buf []byte, l label.Label) {
	w.Write(strconv.AppendUint(buf, uint64(k.From(l)), 10))
}

// Of creates a new Label with this key and the supplied value.
func (k *UInt8) Of(v uint8) label.Label { return label.Of64(k, uint64(v)) }

// Get can be used to get a label for the key from a label.Map.
func (k *UInt8) Get(lm label.Map) uint8 {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return 0
}

// From can be used to get a value from a Label.
func (k *UInt8) From(t label.Label) uint8 { return uint8(t.Unpack64()) }

// UInt16 represents a key
type UInt16 struct {
	name        string
	description string
}

// NewUInt16 creates a new Key for uint16 values.
func NewUInt16(name, description string) *UInt16 {
	return &UInt16{name: name, description: description}
}

func (k *UInt16) Name() string        { return k.name }
func (k *UInt16) Description() string { return k.description }

func (k *UInt16) Format(w io.Writer, buf []byte, l label.Label) {
	w.Write(strconv.AppendUint(buf, uint64(k.From(l)), 10))
}

// Of creates a new Label with this key and the supplied value.
func (k *UInt16) Of(v uint16) label.Label { return label.Of64(k, uint64(v)) }

// Get can be used to get a label for the key from a label.Map.
func (k *UInt16) Get(lm label.Map) uint16 {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return 0
}

// From can be used to get a value from a Label.
func (k *UInt16) From(t label.Label) uint16 { return uint16(t.Unpack64()) }

// UInt32 represents a key
type UInt32 struct {
	name        string
	description string
}

// NewUInt32 creates a new Key for uint32 values.
func NewUInt32(name, description string) *UInt32 {
	return &UInt32{name: name, description: description}
}

func (k *UInt32) Name() string        { return k.name }
func (k *UInt32) Description() string { return k.description }

func (k *UInt32) Format(w io.Writer, buf []byte, l label.Label) {
	w.Write(strconv.AppendUint(buf, uint64(k.From(l)), 10))
}

// Of creates a new Label with this key and the supplied value.
func (k *UInt32) Of(v uint32) label.Label { return label.Of64(k, uint64(v)) }

// Get can be used to get a label for the key from a label.Map.
func (k *UInt32) Get(lm label.Map) uint32 {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return 0
}

// From can be used to get a value from a Label.
func (k *UInt32) From(t label.Label) uint32 { return uint32(t.Unpack64()) }

// UInt64 represents a key
type UInt64 struct {
	name        string
	description string
}

// NewUInt64 creates a new Key for uint64 values.
func NewUInt64(name, description string) *UInt64 {
	return &UInt64{name: name, description: description}
}

func (k *UInt64) Name() string        { return k.name }
func (k *UInt64) Description() string { return k.description }

func (k *UInt64) Format(w io.Writer, buf []byte, l label.Label) {
	w.Write(strconv.AppendUint(buf, k.From(l), 10))
}

// Of creates a new Label with this key and the supplied value.
func (k *UInt64) Of(v uint64) label.Label { return label.Of64(k, v) }

// Get can be used to get a label for the key from a label.Map.
func (k *UInt64) Get(lm label.Map) uint64 {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return 0
}

// From can be used to get a value from a Label.
func (k *UInt64) From(t label.Label) uint64 { return t.Unpack64() }

// Float32 represents a key
type Float32 struct {
	name        string
	description string
}

// NewFloat32 creates a new Key for float32 values.
func NewFloat32(name, description string) *Float32 {
	return &Float32{name: name, description: description}
}

func (k *Float32) Name() string        { return k.name }
func (k *Float32) Description() string { return k.description }

func (k *Float32) Format(w io.Writer, buf []byte, l label.Label) {
	w.Write(strconv.AppendFloat(buf, float64(k.From(l)), 'E', -1, 32))
}

// Of creates a new Label with this key and the supplied value.
func (k *Float32) Of(v float32) label.Label {
	return label.Of64(k, uint64(math.Float32bits(v)))
}

// Get can be used to get a label for the key from a label.Map.
func (k *Float32) Get(lm label.Map) float32 {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return 0
}

// From can be used to get a value from a Label.
func (k *Float32) From(t label.Label) float32 {
	return math.Float32frombits(uint32(t.Unpack64()))
}

// Float64 represents a key
type Float64 struct {
	name        string
	description string
}

// NewFloat64 creates a new Key for int64 values.
func NewFloat64(name, description string) *Float64 {
	return &Float64{name: name, description: description}
}

func (k *Float64) Name() string        { return k.name }
func (k *Float64) Description() string { return k.description }

func (k *Float64) Format(w io.Writer, buf []byte, l label.Label) {
	w.Write(strconv.AppendFloat(buf, k.From(l), 'E', -1, 64))
}

// Of creates a new Label with this key and the supplied value.
func (k *Float64) Of(v float64) label.Label {
	return label.Of64(k, math.Float64bits(v))
}

// Get can be used to get a label for the key from a label.Map.
func (k *Float64) Get(lm label.Map) float64 {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return 0
}

// From can be used to get a value from a Label.
func (k *Float64) From(t label.Label) float64 {
	return math.Float64frombits(t.Unpack64())
}

// String represents a key
type String struct {
	name        string
	description string
}

// NewString creates a new Key for int64 values.
func NewString(name, description string) *String {
	return &String{name: name, description: description}
}

func (k *String) Name() string        { return k.name }
func (k *String) Description() string { return k.description }

func (k *String) Format(w io.Writer, buf []byte, l label.Label) {
	w.Write(strconv.AppendQuote(buf, k.From(l)))
}

// Of creates a new Label with this key and the supplied value.
func (k *String) Of(v string) label.Label { return label.OfString(k, v) }

// Get can be used to get a label for the key from a label.Map.
func (k *String) Get(lm label.Map) string {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return ""
}

// From can be used to get a value from a Label.
func (k *String) From(t label.Label) string { return t.UnpackString() }

// Boolean represents a key
type Boolean struct {
	name        string
	description string
}

// NewBoolean creates a new Key for bool values.
func NewBoolean(name, description string) *Boolean {
	return &Boolean{name: name, description: description}
}

func (k *Boolean) Name() string        { return k.name }
func (k *Boolean) Description() string { return k.description }

func (k *Boolean) Format(w io.Writer, buf []byte, l label.Label) {
	w.Write(strconv.AppendBool(buf, k.From(l)))
}

// Of creates a new Label with this key and the supplied value.
func (k *Boolean) Of(v bool) label.Label {
	if v {
		return label.Of64(k, 1)
	}
	return label.Of64(k, 0)
}

// Get can be used to get a label for the key from a label.Map.
func (k *Boolean) Get(lm label.Map) bool {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return false
}

// From can be used to get a value from a Label.
func (k *Boolean) From(t label.Label) bool { return t.Unpack64() > 0 }

// Error represents a key
type Error struct {
	name        string
	description string
}

// NewError creates a new Key for int64 values.
func NewError(name, description string) *Error {
	return &Error{name: name, description: description}
}

func (k *Error) Name() string        { return k.name }
func (k *Error) Description() string { return k.description }

func (k *Error) Format(w io.Writer, buf []byte, l label.Label) {
	io.WriteString(w, k.From(l).Error())
}

// Of creates a new Label with this key and the supplied value.
func (k *Error) Of(v error) label.Label { return label.OfValue(k, v) }

// Get can be used to get a label for the key from a label.Map.
func (k *Error) Get(lm label.Map) error {
	if t := lm.Find(k); t.Valid() {
		return k.From(t)
	}
	return nil
}

// From can be used to get a value from a Label.
func (k *Error) From(t label.Label) error {
	err, _ := t.UnpackValue().(error)
	return err
}
