// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package event

// Type is the common in-memory representation of the low-leve
type Type uint8

// Spec is a specification for a trace event. It contains sufficient information
// to perform basic parsing of any trace event for any version of Go.
type Spec struct {
	// Name is the human-readable name of the trace event.
	Name string

	// Args contains the names of each trace event's argument.
	// Its length determines the number of arguments an event has.
	//
	// Argument names follow a certain structure and this structure
	// is relied on by the testing framework to type-check arguments.
	// The structure is is:
	//
	//     (?P<name>[A-Za-z]+_)?(?P<type>[A-Za-z]+)
	//
	// In sum, it's an optional name followed by a type. If the name
	// is present, it is separated from the type with an underscore.
	// The valid argument types and the Go types they map to are listed
	// in the ArgTypes variable.
	Args []string

	// StringIDs indicates which of the arguments are string IDs.
	StringIDs []int

	// StackIDs indicates which of the arguments are stack IDs.
	//
	// The list is not sorted. The first index always refers to
	// the main stack for the current execution context of the event.
	StackIDs []int

	// StartEv indicates the event type of the corresponding "start"
	// event, if this event is an "end," for a pair of events that
	// represent a time range.
	StartEv Type

	// IsTimedEvent indicates whether this is an event that both
	// appears in the main event stream and is surfaced to the
	// trace reader.
	//
	// Events that are not "timed" are considered "structural"
	// since they either need significant reinterpretation or
	// otherwise aren't actually surfaced by the trace reader.
	IsTimedEvent bool

	// HasData is true if the event has trailer consisting of a
	// varint length followed by unencoded bytes of some data.
	//
	// An event may not be both a timed event and have data.
	HasData bool

	// IsStack indicates that the event represents a complete
	// stack trace. Specifically, it means that after the arguments
	// there's a varint length, followed by 4*length varints. Each
	// group of 4 represents the PC, file ID, func ID, and line number
	// in that order.
	IsStack bool

	// Experiment indicates the ID of an experiment this event is associated
	// with. If Experiment is not NoExperiment, then the event is experimental
	// and will be exposed as an EventExperiment.
	Experiment Experiment
}

// ArgTypes is a list of valid argument types for use in Args.
//
// See the documentation of Args for more details.
var ArgTypes = [...]string{
	"seq",     // sequence number
	"pstatus", // P status
	"gstatus", // G status
	"g",       // trace.GoID
	"m",       // trace.ThreadID
	"p",       // trace.ProcID
	"string",  // string ID
	"stack",   // stack ID
	"value",   // uint64
	"task",    // trace.TaskID
}

// Names is a helper that produces a mapping of event names to event types.
func Names(specs []Spec) map[string]Type {
	nameToType := make(map[string]Type)
	for i, spec := range specs {
		nameToType[spec.Name] = Type(byte(i))
	}
	return nameToType
}

// Experiment is an experiment ID that events may be associated with.
type Experiment uint

// NoExperiment is the reserved ID 0 indicating no experiment.
const NoExperiment Experiment = 0
