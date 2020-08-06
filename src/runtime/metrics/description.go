// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package metrics

// Description describes a runtime metric.
type Description struct {
	// Name is the full name of the metric which includes the unit.
	//
	// The format of the metric may be described by the following regular expression.
	//
	// 	^(?P<name>/[^:]+):(?P<unit>[^:*/]+(?:[*/][^:*/]+)*)$
	//
	// The format splits the name into two components, separated by a colon: a path which always
	// starts with a /, and a machine-parseable unit. The name may contain any valid Unicode
	// codepoint in between / characters, but by convention will try to stick to lowercase
	// characters and hyphens. An example of such a path might be "/memory/heap/free".
	//
	// The unit is by convention a series of lowercase English unit names (singular or plural)
	// without prefixes delimited by '*' or '/'. The unit names may contain any valid Unicode
	// codepoint that is not a delimiter.
	// Examples of units might be "seconds", "bytes", "bytes/second", "cpu-seconds",
	// "byte*cpu-seconds", and "bytes/second/second".
	//
	// A complete name might look like "/memory/heap/free:bytes".
	Name string

	// Description is an English language sentence describing the metric.
	Description string

	// Kind is the kind of value for this metric.
	//
	// The purpose of this field is to allow users to filter out metrics whose values are
	// types which their application may not understand.
	Kind ValueKind

	// Cumulative is whether or not the metric is cumulative. If a cumulative metric is just
	// a single number, then it increases monotonically. If the metric is a distribution,
	// then each bucket count increases monotonically.
	//
	// This flag thus indicates whether or not it's useful to compute a rate from this value.
	Cumulative bool

	// StopTheWorld is whether or not the metric requires a stop-the-world
	// event in order to collect it.
	StopTheWorld bool
}

// The English language descriptions below must be kept in sync with the
// descriptions of each metric in doc.go.
var allDesc = []Description{
	{
		Name:        "/gc/heap/objects:objects",
		Description: "Number of objects, live or unswept, occupying heap memory.",
		Kind:        KindUint64,
	},
	{
		Name:        "/memory/classes/heap/free:bytes",
		Description: "Memory that is available for allocation, and may be returned to the underlying system.",
		Kind:        KindUint64,
	},
	{
		Name:        "/memory/classes/heap/objects:bytes",
		Description: "Memory occupied by live objects and dead objects that have not yet been collected.",
		Kind:        KindUint64,
	},
	{
		Name:        "/memory/classes/heap/released:bytes",
		Description: "Memory that has been returned to the underlying system.",
		Kind:        KindUint64,
	},
	{
		Name:        "/memory/classes/heap/stacks:bytes",
		Description: "Memory allocated from the heap that is occupied by stacks.",
		Kind:        KindUint64,
	},
	{
		Name:        "/memory/classes/heap/unused:bytes",
		Description: "Memory that is unavailable for allocation, but cannot be returned to the underlying system.",
		Kind:        KindUint64,
	},
	{
		Name:        "/memory/classes/metadata/mcache/free:bytes",
		Description: "Memory that is reserved for runtime mcache structures, but not in-use.",
		Kind:        KindUint64,
	},
	{
		Name:        "/memory/classes/metadata/mcache/inuse:bytes",
		Description: "Memory that is occupied by runtime mcache structures that are currently being used.",
		Kind:        KindUint64,
	},
	{
		Name:        "/memory/classes/metadata/mspan/free:bytes",
		Description: "Memory that is reserved for runtime mspan structures, but not in-use.",
		Kind:        KindUint64,
	},
	{
		Name:        "/memory/classes/metadata/mspan/inuse:bytes",
		Description: "Memory that is occupied by runtime mspan structures that are currently being used.",
		Kind:        KindUint64,
	},
	{
		Name:        "/memory/classes/metadata/other:bytes",
		Description: "Memory that is reserved for or used to hold runtime metadata.",
		Kind:        KindUint64,
	},
	{
		Name:        "/memory/classes/os-stacks:bytes",
		Description: "Stack memory allocated by the underlying operating system.",
		Kind:        KindUint64,
	},
	{
		Name:        "/memory/classes/other:bytes",
		Description: "Memory used by execution trace buffers, structures for debugging the runtime, finalizer and profiler specials, and more.",
		Kind:        KindUint64,
	},
	{
		Name:        "/memory/classes/profiling/buckets:bytes",
		Description: "Memory that is used by the stack trace hash map used for profiling.",
		Kind:        KindUint64,
	},
	{
		Name:        "/memory/classes/total:bytes",
		Description: "All memory mapped by the Go runtime into the current process as read-write. Note that this does not include memory mapped by code called via cgo or via the syscall package. Sum of all metrics in /memory/classes.",
		Kind:        KindUint64,
	},
}

// All returns a slice of containing metric descriptions for all supported metrics.
func All() []Description {
	return allDesc
}
