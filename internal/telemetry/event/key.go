// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

var (
	// Err is a key used to add error values to tag lists.
	Err = NewErrorKey("error", "")
)

// Key is the interface shared by all key implementations.
type Key interface {
	// Name returns the key name.
	Name() string
	// Description returns a string that can be used to describe the value.
	Description() string
}

// key is used as the identity of a Tag.
// Keys are intended to be compared by pointer only, the name should be unique
// for communicating with external systems, but it is not required or enforced.
type key struct {
	name        string
	description string
}

// ValueKey represents a key for untyped values.
type ValueKey struct {
	name        string
	description string
}

// NewKey creates a new Key for untyped values.
func NewKey(name, description string) *ValueKey {
	return &ValueKey{name: name, description: description}
}

func (k *ValueKey) Name() string        { return k.name }
func (k *ValueKey) Description() string { return k.description }

// Get can be used to get a tag for the key from a TagMap.
func (k *ValueKey) Get(tags TagMap) interface{} {
	if t := tags.Find(k); t.Valid() {
		return t.Value
	}
	return nil
}

// Of creates a new Tag with this key and the supplied value.
func (k *ValueKey) Of(value interface{}) Tag { return Tag{Key: k, Value: value} }

// IntKey represents a key
type IntKey struct {
	name        string
	description string
}

// NewIntKey creates a new Key for int values.
func NewIntKey(name, description string) *IntKey {
	return &IntKey{name: name, description: description}
}

func (k *IntKey) Name() string        { return k.name }
func (k *IntKey) Description() string { return k.description }

// Of creates a new Tag with this key and the supplied value.
func (k *IntKey) Of(v int) Tag { return Tag{Key: k, Value: v} }

// Get can be used to get a tag for the key from a TagMap.
func (k *IntKey) Get(tags TagMap) int {
	if t := tags.Find(k); t.Valid() {
		return t.Value.(int)
	}
	return 0
}

// Int8Key represents a key
type Int8Key struct {
	name        string
	description string
}

// NewInt8Key creates a new Key for int8 values.
func NewInt8Key(name, description string) *Int8Key {
	return &Int8Key{name: name, description: description}
}

func (k *Int8Key) Name() string        { return k.name }
func (k *Int8Key) Description() string { return k.description }

// Of creates a new Tag with this key and the supplied value.
func (k *Int8Key) Of(v int8) Tag { return Tag{Key: k, Value: v} }

// Get can be used to get a tag for the key from a TagMap.
func (k *Int8Key) Get(tags TagMap) int8 {
	if t := tags.Find(k); t.Valid() {
		return t.Value.(int8)
	}
	return 0
}

// Int16Key represents a key
type Int16Key struct {
	name        string
	description string
}

// NewInt16Key creates a new Key for int16 values.
func NewInt16Key(name, description string) *Int16Key {
	return &Int16Key{name: name, description: description}
}

func (k *Int16Key) Name() string        { return k.name }
func (k *Int16Key) Description() string { return k.description }

// Of creates a new Tag with this key and the supplied value.
func (k *Int16Key) Of(v int16) Tag { return Tag{Key: k, Value: v} }

// Get can be used to get a tag for the key from a TagMap.
func (k *Int16Key) Get(tags TagMap) int16 {
	if t := tags.Find(k); t.Valid() {
		return t.Value.(int16)
	}
	return 0
}

// Int32Key represents a key
type Int32Key struct {
	name        string
	description string
}

// NewInt32Key creates a new Key for int32 values.
func NewInt32Key(name, description string) *Int32Key {
	return &Int32Key{name: name, description: description}
}

func (k *Int32Key) Name() string        { return k.name }
func (k *Int32Key) Description() string { return k.description }

// Of creates a new Tag with this key and the supplied value.
func (k *Int32Key) Of(v int32) Tag { return Tag{Key: k, Value: v} }

// Get can be used to get a tag for the key from a TagMap.
func (k *Int32Key) Get(tags TagMap) int32 {
	if t := tags.Find(k); t.Valid() {
		return t.Value.(int32)
	}
	return 0
}

// Int64Key represents a key
type Int64Key struct {
	name        string
	description string
}

// NewInt64Key creates a new Key for int64 values.
func NewInt64Key(name, description string) *Int64Key {
	return &Int64Key{name: name, description: description}
}

func (k *Int64Key) Name() string        { return k.name }
func (k *Int64Key) Description() string { return k.description }

// Of creates a new Tag with this key and the supplied value.
func (k *Int64Key) Of(v int64) Tag { return Tag{Key: k, Value: v} }

// Get can be used to get a tag for the key from a TagMap.
func (k *Int64Key) Get(tags TagMap) int64 {
	if t := tags.Find(k); t.Valid() {
		return t.Value.(int64)
	}
	return 0
}

// UIntKey represents a key
type UIntKey struct {
	name        string
	description string
}

// NewUIntKey creates a new Key for uint values.
func NewUIntKey(name, description string) *UIntKey {
	return &UIntKey{name: name, description: description}
}

func (k *UIntKey) Name() string        { return k.name }
func (k *UIntKey) Description() string { return k.description }

// Of creates a new Tag with this key and the supplied value.
func (k *UIntKey) Of(v uint) Tag { return Tag{Key: k, Value: v} }

// Get can be used to get a tag for the key from a TagMap.
func (k *UIntKey) Get(tags TagMap) uint {
	if t := tags.Find(k); t.Valid() {
		return t.Value.(uint)
	}
	return 0
}

// UInt8Key represents a key
type UInt8Key struct {
	name        string
	description string
}

// NewUInt8Key creates a new Key for uint8 values.
func NewUInt8Key(name, description string) *UInt8Key {
	return &UInt8Key{name: name, description: description}
}

func (k *UInt8Key) Name() string        { return k.name }
func (k *UInt8Key) Description() string { return k.description }

// Of creates a new Tag with this key and the supplied value.
func (k *UInt8Key) Of(v uint8) Tag { return Tag{Key: k, Value: v} }

// Get can be used to get a tag for the key from a TagMap.
func (k *UInt8Key) Get(tags TagMap) uint8 {
	if t := tags.Find(k); t.Valid() {
		return t.Value.(uint8)
	}
	return 0
}

// UInt16Key represents a key
type UInt16Key struct {
	name        string
	description string
}

// NewUInt16Key creates a new Key for uint16 values.
func NewUInt16Key(name, description string) *UInt16Key {
	return &UInt16Key{name: name, description: description}
}

func (k *UInt16Key) Name() string        { return k.name }
func (k *UInt16Key) Description() string { return k.description }

// Of creates a new Tag with this key and the supplied value.
func (k *UInt16Key) Of(v uint16) Tag { return Tag{Key: k, Value: v} }

// Get can be used to get a tag for the key from a TagMap.
func (k *UInt16Key) Get(tags TagMap) uint16 {
	if t := tags.Find(k); t.Valid() {
		return t.Value.(uint16)
	}
	return 0
}

// UInt32Key represents a key
type UInt32Key struct {
	name        string
	description string
}

// NewUInt32Key creates a new Key for uint32 values.
func NewUInt32Key(name, description string) *UInt32Key {
	return &UInt32Key{name: name, description: description}
}

func (k *UInt32Key) Name() string        { return k.name }
func (k *UInt32Key) Description() string { return k.description }

// Of creates a new Tag with this key and the supplied value.
func (k *UInt32Key) Of(v uint32) Tag { return Tag{Key: k, Value: v} }

// Get can be used to get a tag for the key from a TagMap.
func (k *UInt32Key) Get(tags TagMap) uint32 {
	if t := tags.Find(k); t.Valid() {
		return t.Value.(uint32)
	}
	return 0
}

// UInt64Key represents a key
type UInt64Key struct {
	name        string
	description string
}

// NewUInt64Key creates a new Key for uint64 values.
func NewUInt64Key(name, description string) *UInt64Key {
	return &UInt64Key{name: name, description: description}
}

func (k *UInt64Key) Name() string        { return k.name }
func (k *UInt64Key) Description() string { return k.description }

// Of creates a new Tag with this key and the supplied value.
func (k *UInt64Key) Of(v uint64) Tag { return Tag{Key: k, Value: v} }

// Get can be used to get a tag for the key from a TagMap.
func (k *UInt64Key) Get(tags TagMap) uint64 {
	if t := tags.Find(k); t.Valid() {
		return t.Value.(uint64)
	}
	return 0
}

// Float32Key represents a key
type Float32Key struct {
	name        string
	description string
}

// NewFloat32Key creates a new Key for float32 values.
func NewFloat32Key(name, description string) *Float32Key {
	return &Float32Key{name: name, description: description}
}

func (k *Float32Key) Name() string        { return k.name }
func (k *Float32Key) Description() string { return k.description }

// Of creates a new Tag with this key and the supplied value.
func (k *Float32Key) Of(v float32) Tag { return Tag{Key: k, Value: v} }

// Get can be used to get a tag for the key from a TagMap.
func (k *Float32Key) Get(tags TagMap) float32 {
	if t := tags.Find(k); t.Valid() {
		return t.Value.(float32)
	}
	return 0
}

// Float64Key represents a key
type Float64Key struct {
	name        string
	description string
}

// NewFloat64Key creates a new Key for int64 values.
func NewFloat64Key(name, description string) *Float64Key {
	return &Float64Key{name: name, description: description}
}

func (k *Float64Key) Name() string        { return k.name }
func (k *Float64Key) Description() string { return k.description }

// Of creates a new Tag with this key and the supplied value.
func (k *Float64Key) Of(v float64) Tag { return Tag{Key: k, Value: v} }

// Get can be used to get a tag for the key from a TagMap.
func (k *Float64Key) Get(tags TagMap) float64 {
	if t := tags.Find(k); t.Valid() {
		return t.Value.(float64)
	}
	return 0
}

// StringKey represents a key
type StringKey struct {
	name        string
	description string
}

// NewStringKey creates a new Key for int64 values.
func NewStringKey(name, description string) *StringKey {
	return &StringKey{name: name, description: description}
}

func (k *StringKey) Name() string        { return k.name }
func (k *StringKey) Description() string { return k.description }

// Of creates a new Tag with this key and the supplied value.
func (k *StringKey) Of(v string) Tag { return Tag{Key: k, Value: v} }

// Get can be used to get a tag for the key from a TagMap.
func (k *StringKey) Get(tags TagMap) string {
	if t := tags.Find(k); t.Valid() {
		return t.Value.(string)
	}
	return ""
}

// BooleanKey represents a key
type BooleanKey struct {
	name        string
	description string
}

// NewBooleanKey creates a new Key for bool values.
func NewBooleanKey(name, description string) *BooleanKey {
	return &BooleanKey{name: name, description: description}
}

func (k *BooleanKey) Name() string        { return k.name }
func (k *BooleanKey) Description() string { return k.description }

// Of creates a new Tag with this key and the supplied value.
func (k *BooleanKey) Of(v bool) Tag { return Tag{Key: k, Value: v} }

// Get can be used to get a tag for the key from a TagMap.
func (k *BooleanKey) Get(tags TagMap) bool {
	if t := tags.Find(k); t.Valid() {
		return t.Value.(bool)
	}
	return false
}

// ErrorKey represents a key
type ErrorKey struct {
	name        string
	description string
}

// NewErrorKey creates a new Key for int64 values.
func NewErrorKey(name, description string) *ErrorKey {
	return &ErrorKey{name: name, description: description}
}

func (k *ErrorKey) Name() string        { return k.name }
func (k *ErrorKey) Description() string { return k.description }

// Of creates a new Tag with this key and the supplied value.
func (k *ErrorKey) Of(v error) Tag { return Tag{Key: k, Value: v} }

// Get can be used to get a tag for the key from a TagMap.
func (k *ErrorKey) Get(tags TagMap) error {
	if t := tags.Find(k); t.Valid() {
		return t.Value.(error)
	}
	return nil
}
