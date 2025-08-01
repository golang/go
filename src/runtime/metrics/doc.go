// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Note: run 'go generate' (which will run 'go test -generate') to update the "Supported metrics" list.
//go:generate go test -run=Docs -generate

/*
Package metrics provides a stable interface to access implementation-defined
metrics exported by the Go runtime. This package is similar to existing functions
like [runtime.ReadMemStats] and [runtime/debug.ReadGCStats], but significantly more general.

The set of metrics defined by this package may evolve as the runtime itself
evolves, and also enables variation across Go implementations, whose relevant
metric sets may not intersect.

# Interface

Metrics are designated by a string key, rather than, for example, a field name in
a struct. The full list of supported metrics is always available in the slice of
Descriptions returned by [All]. Each [Description] also includes useful information
about the metric.

Thus, users of this API are encouraged to sample supported metrics defined by the
slice returned by All to remain compatible across Go versions. Of course, situations
arise where reading specific metrics is critical. For these cases, users are
encouraged to use build tags, and although metrics may be deprecated and removed,
users should consider this to be an exceptional and rare event, coinciding with a
very large change in a particular Go implementation.

Each metric key also has a "kind" (see [ValueKind]) that describes the format of the
metric's value.
In the interest of not breaking users of this package, the "kind" for a given metric
is guaranteed not to change. If it must change, then a new metric will be introduced
with a new key and a new "kind."

# Metric key format

As mentioned earlier, metric keys are strings. Their format is simple and well-defined,
designed to be both human and machine readable. It is split into two components,
separated by a colon: a rooted path and a unit. The choice to include the unit in
the key is motivated by compatibility: if a metric's unit changes, its semantics likely
did also, and a new key should be introduced.

For more details on the precise definition of the metric key's path and unit formats, see
the documentation of the Name field of the Description struct.

# A note about floats

This package supports metrics whose values have a floating-point representation. In
order to improve ease-of-use, this package promises to never produce the following
classes of floating-point values: NaN, infinity.

# Supported metrics

Below is the full list of supported metrics, ordered lexicographically.

	/cgo/go-to-c-calls:calls
		Count of calls made from Go to C by the current process.

	/cpu/classes/gc/mark/assist:cpu-seconds
		Estimated total CPU time goroutines spent performing GC
		tasks to assist the GC and prevent it from falling behind the
		application. This metric is an overestimate, and not directly
		comparable to system CPU time measurements. Compare only with
		other /cpu/classes metrics.

	/cpu/classes/gc/mark/dedicated:cpu-seconds
		Estimated total CPU time spent performing GC tasks on processors
		(as defined by GOMAXPROCS) dedicated to those tasks. This metric
		is an overestimate, and not directly comparable to system CPU
		time measurements. Compare only with other /cpu/classes metrics.

	/cpu/classes/gc/mark/idle:cpu-seconds
		Estimated total CPU time spent performing GC tasks on spare CPU
		resources that the Go scheduler could not otherwise find a use
		for. This should be subtracted from the total GC CPU time to
		obtain a measure of compulsory GC CPU time. This metric is an
		overestimate, and not directly comparable to system CPU time
		measurements. Compare only with other /cpu/classes metrics.

	/cpu/classes/gc/pause:cpu-seconds
		Estimated total CPU time spent with the application paused by
		the GC. Even if only one thread is running during the pause,
		this is computed as GOMAXPROCS times the pause latency because
		nothing else can be executing. This is the exact sum of samples
		in /sched/pauses/total/gc:seconds if each sample is multiplied
		by GOMAXPROCS at the time it is taken. This metric is an
		overestimate, and not directly comparable to system CPU time
		measurements. Compare only with other /cpu/classes metrics.

	/cpu/classes/gc/total:cpu-seconds
		Estimated total CPU time spent performing GC tasks. This metric
		is an overestimate, and not directly comparable to system CPU
		time measurements. Compare only with other /cpu/classes metrics.
		Sum of all metrics in /cpu/classes/gc.

	/cpu/classes/idle:cpu-seconds
		Estimated total available CPU time not spent executing
		any Go or Go runtime code. In other words, the part of
		/cpu/classes/total:cpu-seconds that was unused. This metric is
		an overestimate, and not directly comparable to system CPU time
		measurements. Compare only with other /cpu/classes metrics.

	/cpu/classes/scavenge/assist:cpu-seconds
		Estimated total CPU time spent returning unused memory to the
		underlying platform in response eagerly in response to memory
		pressure. This metric is an overestimate, and not directly
		comparable to system CPU time measurements. Compare only with
		other /cpu/classes metrics.

	/cpu/classes/scavenge/background:cpu-seconds
		Estimated total CPU time spent performing background tasks to
		return unused memory to the underlying platform. This metric is
		an overestimate, and not directly comparable to system CPU time
		measurements. Compare only with other /cpu/classes metrics.

	/cpu/classes/scavenge/total:cpu-seconds
		Estimated total CPU time spent performing tasks that return
		unused memory to the underlying platform. This metric is an
		overestimate, and not directly comparable to system CPU time
		measurements. Compare only with other /cpu/classes metrics.
		Sum of all metrics in /cpu/classes/scavenge.

	/cpu/classes/total:cpu-seconds
		Estimated total available CPU time for user Go code or the Go
		runtime, as defined by GOMAXPROCS. In other words, GOMAXPROCS
		integrated over the wall-clock duration this process has been
		executing for. This metric is an overestimate, and not directly
		comparable to system CPU time measurements. Compare only with
		other /cpu/classes metrics. Sum of all metrics in /cpu/classes.

	/cpu/classes/user:cpu-seconds
		Estimated total CPU time spent running user Go code. This may
		also include some small amount of time spent in the Go runtime.
		This metric is an overestimate, and not directly comparable
		to system CPU time measurements. Compare only with other
		/cpu/classes metrics.

	/gc/cleanups/executed:cleanups
		Approximate total count of cleanup functions (created
		by runtime.AddCleanup) executed by the runtime. Subtract
		/gc/cleanups/queued:cleanups to approximate cleanup queue
		length. Useful for detecting slow cleanups holding up the queue.

	/gc/cleanups/queued:cleanups
		Approximate total count of cleanup functions (created by
		runtime.AddCleanup) queued by the runtime for execution.
		Subtract from /gc/cleanups/executed:cleanups to approximate
		cleanup queue length. Useful for detecting slow cleanups holding
		up the queue.

	/gc/cycles/automatic:gc-cycles
		Count of completed GC cycles generated by the Go runtime.

	/gc/cycles/forced:gc-cycles
		Count of completed GC cycles forced by the application.

	/gc/cycles/total:gc-cycles
		Count of all completed GC cycles.

	/gc/finalizers/executed:finalizers
		Total count of finalizer functions (created by
		runtime.SetFinalizer) executed by the runtime. Subtract
		/gc/finalizers/queued:finalizers to approximate finalizer queue
		length. Useful for detecting finalizers overwhelming the queue,
		either by being too slow, or by there being too many of them.

	/gc/finalizers/queued:finalizers
		Total count of finalizer functions (created by
		runtime.SetFinalizer) and queued by the runtime for execution.
		Subtract from /gc/finalizers/executed:finalizers to approximate
		finalizer queue length. Useful for detecting slow finalizers
		holding up the queue.

	/gc/gogc:percent
		Heap size target percentage configured by the user, otherwise
		100. This value is set by the GOGC environment variable, and the
		runtime/debug.SetGCPercent function.

	/gc/gomemlimit:bytes
		Go runtime memory limit configured by the user, otherwise
		math.MaxInt64. This value is set by the GOMEMLIMIT environment
		variable, and the runtime/debug.SetMemoryLimit function.

	/gc/heap/allocs-by-size:bytes
		Distribution of heap allocations by approximate size.
		Bucket counts increase monotonically. Note that this does not
		include tiny objects as defined by /gc/heap/tiny/allocs:objects,
		only tiny blocks.

	/gc/heap/allocs:bytes
		Cumulative sum of memory allocated to the heap by the
		application.

	/gc/heap/allocs:objects
		Cumulative count of heap allocations triggered by the
		application. Note that this does not include tiny objects as
		defined by /gc/heap/tiny/allocs:objects, only tiny blocks.

	/gc/heap/frees-by-size:bytes
		Distribution of freed heap allocations by approximate size.
		Bucket counts increase monotonically. Note that this does not
		include tiny objects as defined by /gc/heap/tiny/allocs:objects,
		only tiny blocks.

	/gc/heap/frees:bytes
		Cumulative sum of heap memory freed by the garbage collector.

	/gc/heap/frees:objects
		Cumulative count of heap allocations whose storage was freed
		by the garbage collector. Note that this does not include tiny
		objects as defined by /gc/heap/tiny/allocs:objects, only tiny
		blocks.

	/gc/heap/goal:bytes
		Heap size target for the end of the GC cycle.

	/gc/heap/live:bytes
		Heap memory occupied by live objects that were marked by the
		previous GC.

	/gc/heap/objects:objects
		Number of objects, live or unswept, occupying heap memory.

	/gc/heap/tiny/allocs:objects
		Count of small allocations that are packed together into blocks.
		These allocations are counted separately from other allocations
		because each individual allocation is not tracked by the
		runtime, only their block. Each block is already accounted for
		in allocs-by-size and frees-by-size.

	/gc/limiter/last-enabled:gc-cycle
		GC cycle the last time the GC CPU limiter was enabled.
		This metric is useful for diagnosing the root cause of an
		out-of-memory error, because the limiter trades memory for CPU
		time when the GC's CPU time gets too high. This is most likely
		to occur with use of SetMemoryLimit. The first GC cycle is cycle
		1, so a value of 0 indicates that it was never enabled.

	/gc/pauses:seconds
		Deprecated. Prefer the identical /sched/pauses/total/gc:seconds.

	/gc/scan/globals:bytes
		The total amount of global variable space that is scannable.

	/gc/scan/heap:bytes
		The total amount of heap space that is scannable.

	/gc/scan/stack:bytes
		The number of bytes of stack that were scanned last GC cycle.

	/gc/scan/total:bytes
		The total amount space that is scannable. Sum of all metrics in
		/gc/scan.

	/gc/stack/starting-size:bytes
		The stack size of new goroutines.

	/godebug/non-default-behavior/allowmultiplevcs:events
		The number of non-default behaviors executed by the cmd/go
		package due to a non-default GODEBUG=allowmultiplevcs=...
		setting.

	/godebug/non-default-behavior/asynctimerchan:events
		The number of non-default behaviors executed by the time package
		due to a non-default GODEBUG=asynctimerchan=... setting.

	/godebug/non-default-behavior/containermaxprocs:events
		The number of non-default behaviors executed by the runtime
		package due to a non-default GODEBUG=containermaxprocs=...
		setting.

	/godebug/non-default-behavior/embedfollowsymlinks:events
		The number of non-default behaviors executed by the cmd/go
		package due to a non-default GODEBUG=embedfollowsymlinks=...
		setting.

	/godebug/non-default-behavior/execerrdot:events
		The number of non-default behaviors executed by the os/exec
		package due to a non-default GODEBUG=execerrdot=... setting.

	/godebug/non-default-behavior/gocachehash:events
		The number of non-default behaviors executed by the cmd/go
		package due to a non-default GODEBUG=gocachehash=... setting.

	/godebug/non-default-behavior/gocachetest:events
		The number of non-default behaviors executed by the cmd/go
		package due to a non-default GODEBUG=gocachetest=... setting.

	/godebug/non-default-behavior/gocacheverify:events
		The number of non-default behaviors executed by the cmd/go
		package due to a non-default GODEBUG=gocacheverify=... setting.

	/godebug/non-default-behavior/gotestjsonbuildtext:events
		The number of non-default behaviors executed by the cmd/go
		package due to a non-default GODEBUG=gotestjsonbuildtext=...
		setting.

	/godebug/non-default-behavior/gotypesalias:events
		The number of non-default behaviors executed by the go/types
		package due to a non-default GODEBUG=gotypesalias=... setting.

	/godebug/non-default-behavior/http2client:events
		The number of non-default behaviors executed by the net/http
		package due to a non-default GODEBUG=http2client=... setting.

	/godebug/non-default-behavior/http2server:events
		The number of non-default behaviors executed by the net/http
		package due to a non-default GODEBUG=http2server=... setting.

	/godebug/non-default-behavior/httplaxcontentlength:events
		The number of non-default behaviors executed by the net/http
		package due to a non-default GODEBUG=httplaxcontentlength=...
		setting.

	/godebug/non-default-behavior/httpmuxgo121:events
		The number of non-default behaviors executed by the net/http
		package due to a non-default GODEBUG=httpmuxgo121=... setting.

	/godebug/non-default-behavior/httpservecontentkeepheaders:events
		The number of non-default behaviors executed
		by the net/http package due to a non-default
		GODEBUG=httpservecontentkeepheaders=... setting.

	/godebug/non-default-behavior/installgoroot:events
		The number of non-default behaviors executed by the go/build
		package due to a non-default GODEBUG=installgoroot=... setting.

	/godebug/non-default-behavior/multipartmaxheaders:events
		The number of non-default behaviors executed by
		the mime/multipart package due to a non-default
		GODEBUG=multipartmaxheaders=... setting.

	/godebug/non-default-behavior/multipartmaxparts:events
		The number of non-default behaviors executed by
		the mime/multipart package due to a non-default
		GODEBUG=multipartmaxparts=... setting.

	/godebug/non-default-behavior/multipathtcp:events
		The number of non-default behaviors executed by the net package
		due to a non-default GODEBUG=multipathtcp=... setting.

	/godebug/non-default-behavior/netedns0:events
		The number of non-default behaviors executed by the net package
		due to a non-default GODEBUG=netedns0=... setting.

	/godebug/non-default-behavior/panicnil:events
		The number of non-default behaviors executed by the runtime
		package due to a non-default GODEBUG=panicnil=... setting.

	/godebug/non-default-behavior/randautoseed:events
		The number of non-default behaviors executed by the math/rand
		package due to a non-default GODEBUG=randautoseed=... setting.

	/godebug/non-default-behavior/randseednop:events
		The number of non-default behaviors executed by the math/rand
		package due to a non-default GODEBUG=randseednop=... setting.

	/godebug/non-default-behavior/rsa1024min:events
		The number of non-default behaviors executed by the crypto/rsa
		package due to a non-default GODEBUG=rsa1024min=... setting.

	/godebug/non-default-behavior/tarinsecurepath:events
		The number of non-default behaviors executed by the archive/tar
		package due to a non-default GODEBUG=tarinsecurepath=...
		setting.

	/godebug/non-default-behavior/tls10server:events
		The number of non-default behaviors executed by the crypto/tls
		package due to a non-default GODEBUG=tls10server=... setting.

	/godebug/non-default-behavior/tls3des:events
		The number of non-default behaviors executed by the crypto/tls
		package due to a non-default GODEBUG=tls3des=... setting.

	/godebug/non-default-behavior/tlsmaxrsasize:events
		The number of non-default behaviors executed by the crypto/tls
		package due to a non-default GODEBUG=tlsmaxrsasize=... setting.

	/godebug/non-default-behavior/tlsrsakex:events
		The number of non-default behaviors executed by the crypto/tls
		package due to a non-default GODEBUG=tlsrsakex=... setting.

	/godebug/non-default-behavior/tlssha1:events
		The number of non-default behaviors executed by the crypto/tls
		package due to a non-default GODEBUG=tlssha1=... setting.

	/godebug/non-default-behavior/tlsunsafeekm:events
		The number of non-default behaviors executed by the crypto/tls
		package due to a non-default GODEBUG=tlsunsafeekm=... setting.

	/godebug/non-default-behavior/updatemaxprocs:events
		The number of non-default behaviors executed by the runtime
		package due to a non-default GODEBUG=updatemaxprocs=... setting.

	/godebug/non-default-behavior/winreadlinkvolume:events
		The number of non-default behaviors executed by the os package
		due to a non-default GODEBUG=winreadlinkvolume=... setting.

	/godebug/non-default-behavior/winsymlink:events
		The number of non-default behaviors executed by the os package
		due to a non-default GODEBUG=winsymlink=... setting.

	/godebug/non-default-behavior/x509keypairleaf:events
		The number of non-default behaviors executed by the crypto/tls
		package due to a non-default GODEBUG=x509keypairleaf=...
		setting.

	/godebug/non-default-behavior/x509negativeserial:events
		The number of non-default behaviors executed by the crypto/x509
		package due to a non-default GODEBUG=x509negativeserial=...
		setting.

	/godebug/non-default-behavior/x509rsacrt:events
		The number of non-default behaviors executed by the crypto/x509
		package due to a non-default GODEBUG=x509rsacrt=... setting.

	/godebug/non-default-behavior/x509sha256skid:events
		The number of non-default behaviors executed by the crypto/x509
		package due to a non-default GODEBUG=x509sha256skid=... setting.

	/godebug/non-default-behavior/x509usefallbackroots:events
		The number of non-default behaviors executed by the crypto/x509
		package due to a non-default GODEBUG=x509usefallbackroots=...
		setting.

	/godebug/non-default-behavior/x509usepolicies:events
		The number of non-default behaviors executed by the crypto/x509
		package due to a non-default GODEBUG=x509usepolicies=...
		setting.

	/godebug/non-default-behavior/zipinsecurepath:events
		The number of non-default behaviors executed by the archive/zip
		package due to a non-default GODEBUG=zipinsecurepath=...
		setting.

	/memory/classes/heap/free:bytes
		Memory that is completely free and eligible to be returned to
		the underlying system, but has not been. This metric is the
		runtime's estimate of free address space that is backed by
		physical memory.

	/memory/classes/heap/objects:bytes
		Memory occupied by live objects and dead objects that have not
		yet been marked free by the garbage collector.

	/memory/classes/heap/released:bytes
		Memory that is completely free and has been returned to the
		underlying system. This metric is the runtime's estimate of free
		address space that is still mapped into the process, but is not
		backed by physical memory.

	/memory/classes/heap/stacks:bytes
		Memory allocated from the heap that is reserved for stack space,
		whether or not it is currently in-use. Currently, this
		represents all stack memory for goroutines. It also includes all
		OS thread stacks in non-cgo programs. Note that stacks may be
		allocated differently in the future, and this may change.

	/memory/classes/heap/unused:bytes
		Memory that is reserved for heap objects but is not currently
		used to hold heap objects.

	/memory/classes/metadata/mcache/free:bytes
		Memory that is reserved for runtime mcache structures, but not
		in-use.

	/memory/classes/metadata/mcache/inuse:bytes
		Memory that is occupied by runtime mcache structures that are
		currently being used.

	/memory/classes/metadata/mspan/free:bytes
		Memory that is reserved for runtime mspan structures, but not
		in-use.

	/memory/classes/metadata/mspan/inuse:bytes
		Memory that is occupied by runtime mspan structures that are
		currently being used.

	/memory/classes/metadata/other:bytes
		Memory that is reserved for or used to hold runtime metadata.

	/memory/classes/os-stacks:bytes
		Stack memory allocated by the underlying operating system.
		In non-cgo programs this metric is currently zero. This may
		change in the future.In cgo programs this metric includes
		OS thread stacks allocated directly from the OS. Currently,
		this only accounts for one stack in c-shared and c-archive build
		modes, and other sources of stacks from the OS are not measured.
		This too may change in the future.

	/memory/classes/other:bytes
		Memory used by execution trace buffers, structures for debugging
		the runtime, finalizer and profiler specials, and more.

	/memory/classes/profiling/buckets:bytes
		Memory that is used by the stack trace hash map used for
		profiling.

	/memory/classes/total:bytes
		All memory mapped by the Go runtime into the current process
		as read-write. Note that this does not include memory mapped
		by code called via cgo or via the syscall package. Sum of all
		metrics in /memory/classes.

	/sched/gomaxprocs:threads
		The current runtime.GOMAXPROCS setting, or the number of
		operating system threads that can execute user-level Go code
		simultaneously.

	/sched/goroutines:goroutines
		Count of live goroutines.

	/sched/latencies:seconds
		Distribution of the time goroutines have spent in the scheduler
		in a runnable state before actually running. Bucket counts
		increase monotonically.

	/sched/pauses/stopping/gc:seconds
		Distribution of individual GC-related stop-the-world stopping
		latencies. This is the time it takes from deciding to stop the
		world until all Ps are stopped. This is a subset of the total
		GC-related stop-the-world time (/sched/pauses/total/gc:seconds).
		During this time, some threads may be executing. Bucket counts
		increase monotonically.

	/sched/pauses/stopping/other:seconds
		Distribution of individual non-GC-related stop-the-world
		stopping latencies. This is the time it takes from deciding
		to stop the world until all Ps are stopped. This is a
		subset of the total non-GC-related stop-the-world time
		(/sched/pauses/total/other:seconds). During this time, some
		threads may be executing. Bucket counts increase monotonically.

	/sched/pauses/total/gc:seconds
		Distribution of individual GC-related stop-the-world pause
		latencies. This is the time from deciding to stop the world
		until the world is started again. Some of this time is spent
		getting all threads to stop (this is measured directly in
		/sched/pauses/stopping/gc:seconds), during which some threads
		may still be running. Bucket counts increase monotonically.

	/sched/pauses/total/other:seconds
		Distribution of individual non-GC-related stop-the-world
		pause latencies. This is the time from deciding to stop the
		world until the world is started again. Some of this time
		is spent getting all threads to stop (measured directly in
		/sched/pauses/stopping/other:seconds). Bucket counts increase
		monotonically.

	/sync/mutex/wait/total:seconds
		Approximate cumulative time goroutines have spent blocked on a
		sync.Mutex, sync.RWMutex, or runtime-internal lock. This metric
		is useful for identifying global changes in lock contention.
		Collect a mutex or block profile using the runtime/pprof package
		for more detailed contention data.
*/
package metrics
