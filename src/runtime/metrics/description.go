// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package metrics

import "internal/godebugs"

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
	// For histograms, multiple units may apply. For instance, the units of the buckets and
	// the count. By convention, for histograms, the units of the count are always "samples"
	// with the type of sample evident by the metric's name, while the unit in the name
	// specifies the buckets' unit.
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
}

// The English language descriptions below must be kept in sync with the
// descriptions of each metric in doc.go by running 'go generate'.
var allDesc = []Description{
	{
		Name:        "/cgo/go-to-c-calls:calls",
		Description: "Count of calls made from Go to C by the current process.",
		Kind:        KindUint64,
		Cumulative:  true,
	},
	{
		Name: "/cpu/classes/gc/mark/assist:cpu-seconds",
		Description: "Estimated total CPU time goroutines spent performing GC tasks " +
			"to assist the GC and prevent it from falling behind the application. " +
			"This metric is an overestimate, and not directly comparable to " +
			"system CPU time measurements. Compare only with other /cpu/classes " +
			"metrics.",
		Kind:       KindFloat64,
		Cumulative: true,
	},
	{
		Name: "/cpu/classes/gc/mark/dedicated:cpu-seconds",
		Description: "Estimated total CPU time spent performing GC tasks on " +
			"processors (as defined by GOMAXPROCS) dedicated to those tasks. " +
			"This metric is an overestimate, and not directly comparable to " +
			"system CPU time measurements. Compare only with other /cpu/classes " +
			"metrics.",
		Kind:       KindFloat64,
		Cumulative: true,
	},
	{
		Name: "/cpu/classes/gc/mark/idle:cpu-seconds",
		Description: "Estimated total CPU time spent performing GC tasks on " +
			"spare CPU resources that the Go scheduler could not otherwise find " +
			"a use for. This should be subtracted from the total GC CPU time to " +
			"obtain a measure of compulsory GC CPU time. " +
			"This metric is an overestimate, and not directly comparable to " +
			"system CPU time measurements. Compare only with other /cpu/classes " +
			"metrics.",
		Kind:       KindFloat64,
		Cumulative: true,
	},
	{
		Name: "/cpu/classes/gc/pause:cpu-seconds",
		Description: "Estimated total CPU time spent with the application paused by " +
			"the GC. Even if only one thread is running during the pause, this is " +
			"computed as GOMAXPROCS times the pause latency because nothing else " +
			"can be executing. This is the exact sum of samples in " +
			"/sched/pauses/total/gc:seconds if each sample is multiplied by " +
			"GOMAXPROCS at the time it is taken. This metric is an overestimate, " +
			"and not directly comparable to system CPU time measurements. Compare " +
			"only with other /cpu/classes metrics.",
		Kind:       KindFloat64,
		Cumulative: true,
	},
	{
		Name: "/cpu/classes/gc/total:cpu-seconds",
		Description: "Estimated total CPU time spent performing GC tasks. " +
			"This metric is an overestimate, and not directly comparable to " +
			"system CPU time measurements. Compare only with other /cpu/classes " +
			"metrics. Sum of all metrics in /cpu/classes/gc.",
		Kind:       KindFloat64,
		Cumulative: true,
	},
	{
		Name: "/cpu/classes/idle:cpu-seconds",
		Description: "Estimated total available CPU time not spent executing any Go or Go runtime code. " +
			"In other words, the part of /cpu/classes/total:cpu-seconds that was unused. " +
			"This metric is an overestimate, and not directly comparable to " +
			"system CPU time measurements. Compare only with other /cpu/classes " +
			"metrics.",
		Kind:       KindFloat64,
		Cumulative: true,
	},
	{
		Name: "/cpu/classes/scavenge/assist:cpu-seconds",
		Description: "Estimated total CPU time spent returning unused memory to the " +
			"underlying platform in response eagerly in response to memory pressure. " +
			"This metric is an overestimate, and not directly comparable to " +
			"system CPU time measurements. Compare only with other /cpu/classes " +
			"metrics.",
		Kind:       KindFloat64,
		Cumulative: true,
	},
	{
		Name: "/cpu/classes/scavenge/background:cpu-seconds",
		Description: "Estimated total CPU time spent performing background tasks " +
			"to return unused memory to the underlying platform. " +
			"This metric is an overestimate, and not directly comparable to " +
			"system CPU time measurements. Compare only with other /cpu/classes " +
			"metrics.",
		Kind:       KindFloat64,
		Cumulative: true,
	},
	{
		Name: "/cpu/classes/scavenge/total:cpu-seconds",
		Description: "Estimated total CPU time spent performing tasks that return " +
			"unused memory to the underlying platform. " +
			"This metric is an overestimate, and not directly comparable to " +
			"system CPU time measurements. Compare only with other /cpu/classes " +
			"metrics. Sum of all metrics in /cpu/classes/scavenge.",
		Kind:       KindFloat64,
		Cumulative: true,
	},
	{
		Name: "/cpu/classes/total:cpu-seconds",
		Description: "Estimated total available CPU time for user Go code " +
			"or the Go runtime, as defined by GOMAXPROCS. In other words, GOMAXPROCS " +
			"integrated over the wall-clock duration this process has been executing for. " +
			"This metric is an overestimate, and not directly comparable to " +
			"system CPU time measurements. Compare only with other /cpu/classes " +
			"metrics. Sum of all metrics in /cpu/classes.",
		Kind:       KindFloat64,
		Cumulative: true,
	},
	{
		Name: "/cpu/classes/user:cpu-seconds",
		Description: "Estimated total CPU time spent running user Go code. This may " +
			"also include some small amount of time spent in the Go runtime. " +
			"This metric is an overestimate, and not directly comparable to " +
			"system CPU time measurements. Compare only with other /cpu/classes " +
			"metrics.",
		Kind:       KindFloat64,
		Cumulative: true,
	},
	{
		Name: "/gc/cleanups/executed:cleanups",
		Description: "Approximate total count of cleanup functions (created by runtime.AddCleanup) " +
			"executed by the runtime. Subtract /gc/cleanups/queued:cleanups to approximate " +
			"cleanup queue length. Useful for detecting slow cleanups holding up the queue.",
		Kind:       KindUint64,
		Cumulative: true,
	},
	{
		Name: "/gc/cleanups/queued:cleanups",
		Description: "Approximate total count of cleanup functions (created by runtime.AddCleanup) " +
			"queued by the runtime for execution. Subtract from /gc/cleanups/executed:cleanups " +
			"to approximate cleanup queue length. Useful for detecting slow cleanups holding up the queue.",
		Kind:       KindUint64,
		Cumulative: true,
	},
	{
		Name:        "/gc/cycles/automatic:gc-cycles",
		Description: "Count of completed GC cycles generated by the Go runtime.",
		Kind:        KindUint64,
		Cumulative:  true,
	},
	{
		Name:        "/gc/cycles/forced:gc-cycles",
		Description: "Count of completed GC cycles forced by the application.",
		Kind:        KindUint64,
		Cumulative:  true,
	},
	{
		Name:        "/gc/cycles/total:gc-cycles",
		Description: "Count of all completed GC cycles.",
		Kind:        KindUint64,
		Cumulative:  true,
	},
	{
		Name: "/gc/finalizers/executed:finalizers",
		Description: "Total count of finalizer functions (created by runtime.SetFinalizer) " +
			"executed by the runtime. Subtract /gc/finalizers/queued:finalizers to approximate " +
			"finalizer queue length. Useful for detecting finalizers overwhelming the queue, " +
			"either by being too slow, or by there being too many of them.",
		Kind:       KindUint64,
		Cumulative: true,
	},
	{
		Name: "/gc/finalizers/queued:finalizers",
		Description: "Total count of finalizer functions (created by runtime.SetFinalizer) and " +
			"queued by the runtime for execution. Subtract from /gc/finalizers/executed:finalizers " +
			"to approximate finalizer queue length. Useful for detecting slow finalizers holding up the queue.",
		Kind:       KindUint64,
		Cumulative: true,
	},
	{
		Name: "/gc/gogc:percent",
		Description: "Heap size target percentage configured by the user, otherwise 100. This " +
			"value is set by the GOGC environment variable, and the runtime/debug.SetGCPercent " +
			"function.",
		Kind: KindUint64,
	},
	{
		Name: "/gc/gomemlimit:bytes",
		Description: "Go runtime memory limit configured by the user, otherwise " +
			"math.MaxInt64. This value is set by the GOMEMLIMIT environment variable, and " +
			"the runtime/debug.SetMemoryLimit function.",
		Kind: KindUint64,
	},
	{
		Name: "/gc/heap/allocs-by-size:bytes",
		Description: "Distribution of heap allocations by approximate size. " +
			"Bucket counts increase monotonically. " +
			"Note that this does not include tiny objects as defined by " +
			"/gc/heap/tiny/allocs:objects, only tiny blocks.",
		Kind:       KindFloat64Histogram,
		Cumulative: true,
	},
	{
		Name:        "/gc/heap/allocs:bytes",
		Description: "Cumulative sum of memory allocated to the heap by the application.",
		Kind:        KindUint64,
		Cumulative:  true,
	},
	{
		Name: "/gc/heap/allocs:objects",
		Description: "Cumulative count of heap allocations triggered by the application. " +
			"Note that this does not include tiny objects as defined by " +
			"/gc/heap/tiny/allocs:objects, only tiny blocks.",
		Kind:       KindUint64,
		Cumulative: true,
	},
	{
		Name: "/gc/heap/frees-by-size:bytes",
		Description: "Distribution of freed heap allocations by approximate size. " +
			"Bucket counts increase monotonically. " +
			"Note that this does not include tiny objects as defined by " +
			"/gc/heap/tiny/allocs:objects, only tiny blocks.",
		Kind:       KindFloat64Histogram,
		Cumulative: true,
	},
	{
		Name:        "/gc/heap/frees:bytes",
		Description: "Cumulative sum of heap memory freed by the garbage collector.",
		Kind:        KindUint64,
		Cumulative:  true,
	},
	{
		Name: "/gc/heap/frees:objects",
		Description: "Cumulative count of heap allocations whose storage was freed " +
			"by the garbage collector. " +
			"Note that this does not include tiny objects as defined by " +
			"/gc/heap/tiny/allocs:objects, only tiny blocks.",
		Kind:       KindUint64,
		Cumulative: true,
	},
	{
		Name:        "/gc/heap/goal:bytes",
		Description: "Heap size target for the end of the GC cycle.",
		Kind:        KindUint64,
	},
	{
		Name:        "/gc/heap/live:bytes",
		Description: "Heap memory occupied by live objects that were marked by the previous GC.",
		Kind:        KindUint64,
	},
	{
		Name:        "/gc/heap/objects:objects",
		Description: "Number of objects, live or unswept, occupying heap memory.",
		Kind:        KindUint64,
	},
	{
		Name: "/gc/heap/tiny/allocs:objects",
		Description: "Count of small allocations that are packed together into blocks. " +
			"These allocations are counted separately from other allocations " +
			"because each individual allocation is not tracked by the runtime, " +
			"only their block. Each block is already accounted for in " +
			"allocs-by-size and frees-by-size.",
		Kind:       KindUint64,
		Cumulative: true,
	},
	{
		Name: "/gc/limiter/last-enabled:gc-cycle",
		Description: "GC cycle the last time the GC CPU limiter was enabled. " +
			"This metric is useful for diagnosing the root cause of an out-of-memory " +
			"error, because the limiter trades memory for CPU time when the GC's CPU " +
			"time gets too high. This is most likely to occur with use of SetMemoryLimit. " +
			"The first GC cycle is cycle 1, so a value of 0 indicates that it was never enabled.",
		Kind: KindUint64,
	},
	{
		Name:        "/gc/pauses:seconds",
		Description: "Deprecated. Prefer the identical /sched/pauses/total/gc:seconds.",
		Kind:        KindFloat64Histogram,
		Cumulative:  true,
	},
	{
		Name:        "/gc/scan/globals:bytes",
		Description: "The total amount of global variable space that is scannable.",
		Kind:        KindUint64,
	},
	{
		Name:        "/gc/scan/heap:bytes",
		Description: "The total amount of heap space that is scannable.",
		Kind:        KindUint64,
	},
	{
		Name:        "/gc/scan/stack:bytes",
		Description: "The number of bytes of stack that were scanned last GC cycle.",
		Kind:        KindUint64,
	},
	{
		Name:        "/gc/scan/total:bytes",
		Description: "The total amount space that is scannable. Sum of all metrics in /gc/scan.",
		Kind:        KindUint64,
	},
	{
		Name:        "/gc/stack/starting-size:bytes",
		Description: "The stack size of new goroutines.",
		Kind:        KindUint64,
		Cumulative:  false,
	},
	{
		Name: "/memory/classes/heap/free:bytes",
		Description: "Memory that is completely free and eligible to be returned to the underlying system, " +
			"but has not been. This metric is the runtime's estimate of free address space that is backed by " +
			"physical memory.",
		Kind: KindUint64,
	},
	{
		Name:        "/memory/classes/heap/objects:bytes",
		Description: "Memory occupied by live objects and dead objects that have not yet been marked free by the garbage collector.",
		Kind:        KindUint64,
	},
	{
		Name: "/memory/classes/heap/released:bytes",
		Description: "Memory that is completely free and has been returned to the underlying system. This " +
			"metric is the runtime's estimate of free address space that is still mapped into the process, " +
			"but is not backed by physical memory.",
		Kind: KindUint64,
	},
	{
		Name: "/memory/classes/heap/stacks:bytes",
		Description: "Memory allocated from the heap that is reserved for stack space, whether or not it is currently in-use. " +
			"Currently, this represents all stack memory for goroutines. It also includes all OS thread stacks in non-cgo programs. " +
			"Note that stacks may be allocated differently in the future, and this may change.",
		Kind: KindUint64,
	},
	{
		Name:        "/memory/classes/heap/unused:bytes",
		Description: "Memory that is reserved for heap objects but is not currently used to hold heap objects.",
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
		Name: "/memory/classes/os-stacks:bytes",
		Description: "Stack memory allocated by the underlying operating system. " +
			"In non-cgo programs this metric is currently zero. This may change in the future." +
			"In cgo programs this metric includes OS thread stacks allocated directly from the OS. " +
			"Currently, this only accounts for one stack in c-shared and c-archive build modes, " +
			"and other sources of stacks from the OS are not measured. This too may change in the future.",
		Kind: KindUint64,
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
	{
		Name:        "/sched/gomaxprocs:threads",
		Description: "The current runtime.GOMAXPROCS setting, or the number of operating system threads that can execute user-level Go code simultaneously.",
		Kind:        KindUint64,
	},
	{
		Name:        "/sched/goroutines-created:goroutines",
		Description: "Count of goroutines created since program start.",
		Cumulative:  true,
		Kind:        KindUint64,
	},
	{
		Name:        "/sched/goroutines/not-in-go:goroutines",
		Description: "Approximate count of goroutines running or blocked in a system call or cgo call. Not guaranteed to add up to /sched/goroutines:goroutines with other goroutine metrics.",
		Kind:        KindUint64,
	},
	{
		Name:        "/sched/goroutines/runnable:goroutines",
		Description: "Approximate count of goroutines ready to execute, but not executing. Not guaranteed to add up to /sched/goroutines:goroutines with other goroutine metrics.",
		Kind:        KindUint64,
	},
	{
		Name:        "/sched/goroutines/running:goroutines",
		Description: "Approximate count of goroutines executing. Always less than or equal to /sched/gomaxprocs:threads. Not guaranteed to add up to /sched/goroutines:goroutines with other goroutine metrics.",
		Kind:        KindUint64,
	},
	{
		Name:        "/sched/goroutines/waiting:goroutines",
		Description: "Approximate count of goroutines waiting on a resource (I/O or sync primitives). Not guaranteed to add up to /sched/goroutines:goroutines with other goroutine metrics.",
		Kind:        KindUint64,
	},
	{
		Name:        "/sched/goroutines:goroutines",
		Description: "Count of live goroutines.",
		Kind:        KindUint64,
	},
	{
		Name:        "/sched/latencies:seconds",
		Description: "Distribution of the time goroutines have spent in the scheduler in a runnable state before actually running. Bucket counts increase monotonically.",
		Kind:        KindFloat64Histogram,
		Cumulative:  true,
	},
	{
		Name:        "/sched/pauses/stopping/gc:seconds",
		Description: "Distribution of individual GC-related stop-the-world stopping latencies. This is the time it takes from deciding to stop the world until all Ps are stopped. This is a subset of the total GC-related stop-the-world time (/sched/pauses/total/gc:seconds). During this time, some threads may be executing. Bucket counts increase monotonically.",
		Kind:        KindFloat64Histogram,
		Cumulative:  true,
	},
	{
		Name:        "/sched/pauses/stopping/other:seconds",
		Description: "Distribution of individual non-GC-related stop-the-world stopping latencies. This is the time it takes from deciding to stop the world until all Ps are stopped. This is a subset of the total non-GC-related stop-the-world time (/sched/pauses/total/other:seconds). During this time, some threads may be executing. Bucket counts increase monotonically.",
		Kind:        KindFloat64Histogram,
		Cumulative:  true,
	},
	{
		Name:        "/sched/pauses/total/gc:seconds",
		Description: "Distribution of individual GC-related stop-the-world pause latencies. This is the time from deciding to stop the world until the world is started again. Some of this time is spent getting all threads to stop (this is measured directly in /sched/pauses/stopping/gc:seconds), during which some threads may still be running. Bucket counts increase monotonically.",
		Kind:        KindFloat64Histogram,
		Cumulative:  true,
	},
	{
		Name:        "/sched/pauses/total/other:seconds",
		Description: "Distribution of individual non-GC-related stop-the-world pause latencies. This is the time from deciding to stop the world until the world is started again. Some of this time is spent getting all threads to stop (measured directly in /sched/pauses/stopping/other:seconds). Bucket counts increase monotonically.",
		Kind:        KindFloat64Histogram,
		Cumulative:  true,
	},
	{
		Name:        "/sched/threads/total:threads",
		Description: "The current count of live threads that are owned by the Go runtime.",
		Kind:        KindUint64,
	},
	{
		Name:        "/sync/mutex/wait/total:seconds",
		Description: "Approximate cumulative time goroutines have spent blocked on a sync.Mutex, sync.RWMutex, or runtime-internal lock. This metric is useful for identifying global changes in lock contention. Collect a mutex or block profile using the runtime/pprof package for more detailed contention data.",
		Kind:        KindFloat64,
		Cumulative:  true,
	},
}

func init() {
	// Insert all the non-default-reporting GODEBUGs into the table,
	// preserving the overall sort order.
	i := 0
	for i < len(allDesc) && allDesc[i].Name < "/godebug/" {
		i++
	}
	more := make([]Description, i, len(allDesc)+len(godebugs.All))
	copy(more, allDesc)
	for _, info := range godebugs.All {
		if !info.Opaque {
			more = append(more, Description{
				Name: "/godebug/non-default-behavior/" + info.Name + ":events",
				Description: "The number of non-default behaviors executed by the " +
					info.Package + " package " + "due to a non-default " +
					"GODEBUG=" + info.Name + "=... setting.",
				Kind:       KindUint64,
				Cumulative: true,
			})
		}
	}
	allDesc = append(more, allDesc[i:]...)
}

// All returns a slice of containing metric descriptions for all supported metrics.
func All() []Description {
	return allDesc
}
