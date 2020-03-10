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
	// Identity returns the underlying key identity.
	Identity() interface{}
	// Name returns the key name.
	Name() string
	// Description returns a string that can be used to describe the value.
	Description() string
	// OfValue creates a new Tag with this key and the supplied untyped value.
	OfValue(interface{}) Tag
}

// key is used as the identity of a Tag.
// Keys are intended to be compared by pointer only, the name should be unique
// for communicating with external systems, but it is not required or enforced.
type key struct {
	name        string
	description string
}

func newKey(name, description string) *key   { return &key{name: name, description: description} }
func (k *key) Name() string                  { return k.name }
func (k *key) Description() string           { return k.description }
func (k *key) Identity() interface{}         { return k }
func (k *key) OfValue(value interface{}) Tag { return Tag{key: k, value: value} }

// ValueKey represents a key for untyped values.
type ValueKey struct{ *key }

// NewKey creates a new Key for untyped values.
func NewKey(name, description string) ValueKey {
	return ValueKey{newKey(name, description)}
}

// Get can be used to get a tag for the key from a TagList.
func (k ValueKey) Get(tags TagSet) interface{} {
	if t := tags.find(k.key); t.key != nil {
		return t.value
	}
	return nil
}

// Of creates a new Tag with this key and the supplied value.
func (k ValueKey) Of(value interface{}) Tag { return Tag{key: k.key, value: value} }

// IntKey represents a key
type IntKey struct{ *key }

// NewIntKey creates a new Key for int values.
func NewIntKey(name, description string) IntKey {
	return IntKey{newKey(name, description)}
}

// Of creates a new Tag with this key and the supplied value.
func (k IntKey) Of(v int) Tag { return Tag{key: k.key, value: v} }

// Get can be used to get a tag for the key from a TagSet.
func (k IntKey) Get(tags TagSet) int {
	if t := tags.find(k.key); t.key != nil {
		return t.value.(int)
	}
	return 0
}

// Int8Key represents a key
type Int8Key struct{ *key }

// NewInt8Key creates a new Key for int8 values.
func NewInt8Key(name, description string) Int8Key {
	return Int8Key{newKey(name, description)}
}

// Of creates a new Tag with this key and the supplied value.
func (k Int8Key) Of(v int8) Tag { return Tag{key: k.key, value: v} }

// Get can be used to get a tag for the key from a TagSet.
func (k Int8Key) Get(tags TagSet) int8 {
	if t := tags.find(k.key); t.key != nil {
		return t.value.(int8)
	}
	return 0
}

// Int16Key represents a key
type Int16Key struct{ *key }

// NewInt16Key creates a new Key for int16 values.
func NewInt16Key(name, description string) Int16Key {
	return Int16Key{newKey(name, description)}
}

// Of creates a new Tag with this key and the supplied value.
func (k Int16Key) Of(v int16) Tag { return Tag{key: k.key, value: v} }

// Get can be used to get a tag for the key from a TagSet.
func (k Int16Key) Get(tags TagSet) int16 {
	if t := tags.find(k.key); t.key != nil {
		return t.value.(int16)
	}
	return 0
}

// Int32Key represents a key
type Int32Key struct{ *key }

// NewInt32Key creates a new Key for int32 values.
func NewInt32Key(name, description string) Int32Key {
	return Int32Key{newKey(name, description)}
}

// Of creates a new Tag with this key and the supplied value.
func (k Int32Key) Of(v int32) Tag { return Tag{key: k.key, value: v} }

// Get can be used to get a tag for the key from a TagSet.
func (k Int32Key) Get(tags TagSet) int32 {
	if t := tags.find(k.key); t.key != nil {
		return t.value.(int32)
	}
	return 0
}

// Int64Key represents a key
type Int64Key struct{ *key }

// NewInt64Key creates a new Key for int64 values.
func NewInt64Key(name, description string) Int64Key {
	return Int64Key{newKey(name, description)}
}

// Of creates a new Tag with this key and the supplied value.
func (k Int64Key) Of(v int64) Tag { return Tag{key: k.key, value: v} }

// Get can be used to get a tag for the key from a TagSet.
func (k Int64Key) Get(tags TagSet) int64 {
	if t := tags.find(k.key); t.key != nil {
		return t.value.(int64)
	}
	return 0
}

// UIntKey represents a key
type UIntKey struct{ *key }

// NewUIntKey creates a new Key for uint values.
func NewUIntKey(name, description string) UIntKey {
	return UIntKey{newKey(name, description)}
}

// Of creates a new Tag with this key and the supplied value.
func (k UIntKey) Of(v uint) Tag { return Tag{key: k.key, value: v} }

// Get can be used to get a tag for the key from a TagSet.
func (k UIntKey) Get(tags TagSet) uint {
	if t := tags.find(k.key); t.key != nil {
		return t.value.(uint)
	}
	return 0
}

// UInt8Key represents a key
type UInt8Key struct{ *key }

// NewUInt8Key creates a new Key for uint8 values.
func NewUInt8Key(name, description string) UInt8Key {
	return UInt8Key{newKey(name, description)}
}

// Of creates a new Tag with this key and the supplied value.
func (k UInt8Key) Of(v uint8) Tag { return Tag{key: k.key, value: v} }

// Get can be used to get a tag for the key from a TagSet.
func (k UInt8Key) Get(tags TagSet) uint8 {
	if t := tags.find(k.key); t.key != nil {
		return t.value.(uint8)
	}
	return 0
}

// UInt16Key represents a key
type UInt16Key struct{ *key }

// NewUInt16Key creates a new Key for uint16 values.
func NewUInt16Key(name, description string) UInt16Key {
	return UInt16Key{newKey(name, description)}
}

// Of creates a new Tag with this key and the supplied value.
func (k UInt16Key) Of(v uint16) Tag { return Tag{key: k.key, value: v} }

// Get can be used to get a tag for the key from a TagSet.
func (k UInt16Key) Get(tags TagSet) uint16 {
	if t := tags.find(k.key); t.key != nil {
		return t.value.(uint16)
	}
	return 0
}

// UInt32Key represents a key
type UInt32Key struct{ *key }

// NewUInt32Key creates a new Key for uint32 values.
func NewUInt32Key(name, description string) UInt32Key {
	return UInt32Key{newKey(name, description)}
}

// Of creates a new Tag with this key and the supplied value.
func (k UInt32Key) Of(v uint32) Tag { return Tag{key: k.key, value: v} }

// Get can be used to get a tag for the key from a TagSet.
func (k UInt32Key) Get(tags TagSet) uint32 {
	if t := tags.find(k.key); t.key != nil {
		return t.value.(uint32)
	}
	return 0
}

// UInt64Key represents a key
type UInt64Key struct{ *key }

// NewUInt64Key creates a new Key for uint64 values.
func NewUInt64Key(name, description string) UInt64Key {
	return UInt64Key{newKey(name, description)}
}

// Of creates a new Tag with this key and the supplied value.
func (k UInt64Key) Of(v uint64) Tag { return Tag{key: k.key, value: v} }

// Get can be used to get a tag for the key from a TagSet.
func (k UInt64Key) Get(tags TagSet) uint64 {
	if t := tags.find(k.key); t.key != nil {
		return t.value.(uint64)
	}
	return 0
}

// Float32Key represents a key
type Float32Key struct{ *key }

// NewFloat32Key creates a new Key for float32 values.
func NewFloat32Key(name, description string) Float32Key {
	return Float32Key{newKey(name, description)}
}

// Of creates a new Tag with this key and the supplied value.
func (k Float32Key) Of(v float32) Tag { return Tag{key: k.key, value: v} }

// Get can be used to get a tag for the key from a TagSet.
func (k Float32Key) Get(tags TagSet) float32 {
	if t := tags.find(k.key); t.key != nil {
		return t.value.(float32)
	}
	return 0
}

// Float64Key represents a key
type Float64Key struct{ *key }

// NewFloat64Key creates a new Key for int64 values.
func NewFloat64Key(name, description string) Float64Key {
	return Float64Key{newKey(name, description)}
}

// Of creates a new Tag with this key and the supplied value.
func (k Float64Key) Of(v float64) Tag { return Tag{key: k.key, value: v} }

// Get can be used to get a tag for the key from a TagSet.
func (k Float64Key) Get(tags TagSet) float64 {
	if t := tags.find(k.key); t.key != nil {
		return t.value.(float64)
	}
	return 0
}

// StringKey represents a key
type StringKey struct{ *key }

// NewStringKey creates a new Key for int64 values.
func NewStringKey(name, description string) StringKey {
	return StringKey{newKey(name, description)}
}

// Of creates a new Tag with this key and the supplied value.
func (k StringKey) Of(v string) Tag { return Tag{key: k.key, value: v} }

// Get can be used to get a tag for the key from a TagSet.
func (k StringKey) Get(tags TagSet) string {
	if t := tags.find(k.key); t.key != nil {
		return t.value.(string)
	}
	return ""
}

// BooleanKey represents a key
type BooleanKey struct{ *key }

// NewBooleanKey creates a new Key for bool values.
func NewBooleanKey(name, description string) BooleanKey {
	return BooleanKey{newKey(name, description)}
}

// Of creates a new Tag with this key and the supplied value.
func (k BooleanKey) Of(v bool) Tag { return Tag{key: k.key, value: v} }

// Get can be used to get a tag for the key from a TagSet.
func (k BooleanKey) Get(tags TagSet) bool {
	if t := tags.find(k.key); t.key != nil {
		return t.value.(bool)
	}
	return false
}

// ErrorKey represents a key
type ErrorKey struct{ *key }

// NewErrorKey creates a new Key for int64 values.
func NewErrorKey(name, description string) ErrorKey {
	return ErrorKey{newKey(name, description)}
}

// Of creates a new Tag with this key and the supplied value.
func (k ErrorKey) Of(v error) Tag { return Tag{key: k.key, value: v} }

// Get can be used to get a tag for the key from a TagSet.
func (k ErrorKey) Get(tags TagSet) error {
	if t := tags.find(k.key); t.key != nil {
		return t.value.(error)
	}
	return nil
}
