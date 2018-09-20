<!--{
	"Title": "Diagnostics",
	"Template": true
}-->

<!--
NOTE: In this document and others in this directory, the convention is to
set fixed-width phrases with non-fixed-width spaces, as in
<code>hello</code> <code>world</code>.
Do not send CLs removing the interior tags from such phrases.
-->

<h2 id="introduction">Introduction</h2>

<p>
The Go ecosystem provides a large suite of APIs and tools to
diagnose logic and performance problems in Go programs. This page
summarizes the available tools and helps Go users pick the right one
for their specific problem.
</p>

<p>
Diagnostics solutions can be categorized into the following groups:
</p>

<ul>
<li><strong>Profiling</strong>: Profiling tools analyze the complexity and costs of a
Go program such as its memory usage and frequently called
functions to identify the expensive sections of a Go program.</li>
<li><strong>Tracing</strong>: Tracing is a way to instrument code to analyze latency
throughout the lifecycle of a call or user request. Traces provide an
overview of how much latency each component contributes to the overall
latency in a system. Traces can span multiple Go processes.</li>
<li><strong>Debugging</strong>: Debugging allows us to pause a Go program and examine
its execution. Program state and flow can be verified with debugging.</li>
<li><strong>Runtime statistics and events</strong>: Collection and analysis of runtime stats and events
provides a high-level overview of the health of Go programs. Spikes/dips of metrics
helps us to identify changes in throughput, utilization, and performance.</li>
</ul>

<p>
Note: Some diagnostics tools may interfere with each other. For example, precise
memory profiling skews CPU profiles and goroutine blocking profiling affects scheduler
trace. Use tools in isolation to get more precise info.
</p>

<h2 id="profiling">Profiling</h2>

<p>
Profiling is useful for identifying expensive or frequently called sections
of code. The Go runtime provides <a href="https://golang.org/pkg/runtime/pprof/">
profiling data</a> in the format expected by the
<a href="https://github.com/google/pprof/blob/master/doc/README.md">pprof visualization tool</a>.
The profiling data can be collected during testing
via <code>go</code> <code>test</code> or endpoints made available from the <a href="/pkg/net/http/pprof/">
net/http/pprof</a> package. Users need to collect the profiling data and use pprof tools to filter
and visualize the top code paths.
</p>

<p>Predefined profiles provided by the <a href="/pkg/runtime/pprof">runtime/pprof</a> package:</p>

<ul>
<li>
<strong>cpu</strong>: CPU profile determines where a program spends
its time while actively consuming CPU cycles (as opposed to while sleeping or waiting for I/O).
</li>
<li>
<strong>heap</strong>: Heap profile reports memory allocation samples;
used to monitor current and historical memory usage, and to check for memory leaks.
</li>
<li>
<strong>threadcreate</strong>: Thread creation profile reports the sections
of the program that lead the creation of new OS threads.
</li>
<li>
<strong>goroutine</strong>: Goroutine profile reports the stack traces of all current goroutines.
</li>
<li>
<strong>block</strong>: Block profile shows where goroutines block waiting on synchronization
primitives (including timer channels). Block profile is not enabled by default;
use <code>runtime.SetBlockProfileRate</code> to enable it.
</li>
<li>
<strong>mutex</strong>: Mutex profile reports the lock contentions. When you think your
CPU is not fully utilized due to a mutex contention, use this profile. Mutex profile
is not enabled by default, see <code>runtime.SetMutexProfileFraction</code> to enable it.
</li>
</ul>


<p><strong>What other profilers can I use to profile Go programs?</strong></p>

<p>
On Linux, <a href="https://perf.wiki.kernel.org/index.php/Tutorial">perf tools</a>
can be used for profiling Go programs. Perf can profile
and unwind cgo/SWIG code and kernel, so it can be useful to get insights into
native/kernel performance bottlenecks. On macOS,
<a href="https://developer.apple.com/library/content/documentation/DeveloperTools/Conceptual/InstrumentsUserGuide/">Instruments</a>
suite can be used profile Go programs.
</p>

<p><strong>Can I profile my production services?</strong></p>

<p>Yes. It is safe to profile programs in production, but enabling
some profiles (e.g. the CPU profile) adds cost. You should expect to
see performance downgrade. The performance penalty can be estimated
by measuring the overhead of the profiler before turning it on in
production.
</p>

<p>
You may want to periodically profile your production services.
Especially in a system with many replicas of a single process, selecting
a random replica periodically is a safe option.
Select a production process, profile it for
X seconds for every Y seconds and save the results for visualization and
analysis; then repeat periodically. Results may be manually and/or automatically
reviewed to find problems.
Collection of profiles can interfere with each other,
so it is recommended to collect only a single profile at a time.
</p>

<p>
<strong>What are the best ways to visualize the profiling data?</strong>
</p>

<p>
The Go tools provide text, graph, and <a href="http://valgrind.org/docs/manual/cl-manual.html">callgrind</a>
visualization of the profile data using
<code><a href="https://github.com/google/pprof/blob/master/doc/README.md">go tool pprof</a></code>.
Read <a href="https://blog.golang.org/profiling-go-programs">Profiling Go programs</a>
to see them in action.
</p>

<p>
<img width="800" src="https://storage.googleapis.com/golangorg-assets/pprof-text.png">
<br>
<small>Listing of the most expensive calls as text.</small>
</p>

<p>
<img width="800" src="https://storage.googleapis.com/golangorg-assets/pprof-dot.png">
<br>
<small>Visualization of the most expensive calls as a graph.</small>
</p>

<p>Weblist view displays the expensive parts of the source line by line in
an HTML page. In the following example, 530ms is spent in the
<code>runtime.concatstrings</code> and cost of each line is presented
in the listing.</p>

<p>
<img width="800" src="https://storage.googleapis.com/golangorg-assets/pprof-weblist.png">
<br>
<small>Visualization of the most expensive calls as weblist.</small>
</p>

<p>
Another way to visualize profile data is a <a href="http://www.brendangregg.com/flamegraphs.html">flame graph</a>.
Flame graphs allow you to move in a specific ancestry path, so you can zoom
in/out of specific sections of code.
The <a href="https://github.com/google/pprof">upstream pprof</a>
has support for flame graphs.
</p>

<p>
<img width="800" src="https://storage.googleapis.com/golangorg-assets/flame.png">
<br>
<small>Flame graphs offers visualization to spot the most expensive code-paths.</small>
</p>

<p><strong>Am I restricted to the built-in profiles?</strong></p>

<p>
Additionally to what is provided by the runtime, Go users can create
their custom profiles via <a href="/pkg/runtime/pprof/#Profile">pprof.Profile</a>
and use the existing tools to examine them.
</p>

<p><strong>Can I serve the profiler handlers (/debug/pprof/...) on a different path and port?</strong></p>

<p>
Yes. The <code>net/http/pprof</code> package registers its handlers to the default
mux by default, but you can also register them yourself by using the handlers
exported from the package.
</p>

<p>
For example, the following example will serve the pprof.Profile
handler on :7777 at /custom_debug_path/profile:
</p>

<p>
<pre>
package main

import (
	"log"
	"net/http"
	"net/http/pprof"
)

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/custom_debug_path/profile", pprof.Profile)
	log.Fatal(http.ListenAndServe(":7777", mux))
}
</pre>
</p>

<h2 id="tracing">Tracing</h2>

<p>
Tracing is a way to instrument code to analyze latency throughout the
lifecycle of a chain of calls. Go provides
<a href="https://godoc.org/golang.org/x/net/trace">golang.org/x/net/trace</a>
package as a minimal tracing backend per Go node and provides a minimal
instrumentation library with a simple dashboard. Go also provides
an execution tracer to trace the runtime events within an interval.
</p>

<p>Tracing enables us to:</p>

<ul>
<li>Instrument and analyze application latency in a Go process.</li>
<li>Measure the cost of specific calls in a long chain of calls.</li>
<li>Figure out the utilization and performance improvements.
Bottlenecks are not always obvious without tracing data.</li>
</ul>

<p>
In monolithic systems, it's relatively easy to collect diagnostic data
from the building blocks of a program. All modules live within one
process and share common resources to report logs, errors, and other
diagnostic information. Once your system grows beyond a single process and
starts to become distributed, it becomes harder to follow a call starting
from the front-end web server to all of its back-ends until a response is
returned back to the user. This is where distributed tracing plays a big
role to instrument and analyze your production systems.
</p>

<p>
Distributed tracing is a way to instrument code to analyze latency throughout
the lifecycle of a user request. When a system is distributed and when
conventional profiling and debugging tools don’t scale, you might want
to use distributed tracing tools to analyze the performance of your user
requests and RPCs.
</p>

<p>Distributed tracing enables us to:</p>

<ul>
<li>Instrument and profile application latency in a large system.</li>
<li>Track all RPCs within the lifecycle of a user request and see integration issues
that are only visible in production.</li>
<li>Figure out performance improvements that can be applied to our systems.
Many bottlenecks are not obvious before the collection of tracing data.</li>
</ul>

<p>The Go ecosystem provides various distributed tracing libraries per tracing system
and backend-agnostic ones.</p>


<p><strong>Is there a way to automatically intercept each function call and create traces?</strong></p>

<p>
Go doesn’t provide a way to automatically intercept every function call and create
trace spans. You need to manually instrument your code to create, end, and annotate spans.
</p>

<p><strong>How should I propagate trace headers in Go libraries?</strong></p>

<p>
You can propagate trace identifiers and tags in the
<a href="/pkg/context#Context"><code>context.Context</code></a>.
There is no canonical trace key or common representation of trace headers
in the industry yet. Each tracing provider is responsible for providing propagation
utilities in their Go libraries.
</p>

<p>
<strong>What other low-level events from the standard library or
runtime can be included in a trace?</strong>
</p>

<p>
The standard library and runtime are trying to expose several additional APIs
to notify on low level internal events. For example,
<a href="/pkg/net/http/httptrace#ClientTrace"><code>httptrace.ClientTrace</code></a>
provides APIs to follow low-level events in the life cycle of an outgoing request.
There is an ongoing effort to retrieve low-level runtime events from
the runtime execution tracer and allow users to define and record their user events.
</p>

<h2 id="debugging">Debugging</h2>

<p>
Debugging is the process of identifying why a program misbehaves.
Debuggers allow us to understand a program’s execution flow and current state.
There are several styles of debugging; this section will only focus on attaching
a debugger to a program and core dump debugging.
</p>

<p>Go users mostly use the following debuggers:</p>

<ul>
<li>
<a href="https://github.com/derekparker/delve">Delve</a>:
Delve is a debugger for the Go programming language. It has
support for Go’s runtime concepts and built-in types. Delve is
trying to be a fully featured reliable debugger for Go programs.
</li>
<li>
<a href="https://golang.org/doc/gdb">GDB</a>:
Go provides GDB support via the standard Go compiler and Gccgo.
The stack management, threading, and runtime contain aspects that differ
enough from the execution model GDB expects that they can confuse the
debugger, even when the program is compiled with gccgo. Even though
GDB can be used to debug Go programs, it is not ideal and may
create confusion.
</li>
</ul>

<p><strong>How well do debuggers work with Go programs?</strong></p>

<p>
The <code>gc</code> compiler performs optimizations such as
function inlining and variable registerization. These optimizations
sometimes make debugging with debuggers harder. There is an ongoing
effort to improve the quality of the DWARF information generated for
optimized binaries. Until those improvements are available, we recommend
disabling optimizations when building the code being debugged. The following
command builds a package with no compiler optimizations:

<p>
<pre>
$ go build -gcflags=all="-N -l"
</pre>
</p>

As part of the improvement effort, Go 1.10 introduced a new compiler
flag <code>-dwarflocationlists</code>. The flag causes the compiler to
add location lists that helps debuggers work with optimized binaries.
The following command builds a package with optimizations but with
the DWARF location lists:

<p>
<pre>
$ go build -gcflags="-dwarflocationlists=true"
</pre>
</p>

<p><strong>What’s the recommended debugger user interface?</strong></p>

<p>
Even though both delve and gdb provides CLIs, most editor integrations
and IDEs provides debugging-specific user interfaces.
</p>

<p><strong>Is it possible to do postmortem debugging with Go programs?</strong></p>

<p>
A core dump file is a file that contains the memory dump of a running
process and its process status. It is primarily used for post-mortem
debugging of a program and to understand its state
while it is still running. These two cases make debugging of core
dumps a good diagnostic aid to postmortem and analyze production
services. It is possible to obtain core files from Go programs and
use delve or gdb to debug, see the
<a href="https://golang.org/wiki/CoreDumpDebugging">core dump debugging</a>
page for a step-by-step guide.
</p>

<h2 id="runtime">Runtime statistics and events</h2>

<p>
The runtime provides stats and reporting of internal events for
users to diagnose performance and utilization problems at the
runtime level.
</p>

<p>
Users can monitor these stats to better understand the overall
health and performance of Go programs.
Some frequently monitored stats and states:
</p>

<ul>
<li><code><a href="/pkg/runtime/#ReadMemStats">runtime.ReadMemStats</a></code>
reports the metrics related to heap
allocation and garbage collection. Memory stats are useful for
monitoring how much memory resources a process is consuming,
whether the process can utilize memory well, and to catch
memory leaks.</li>
<li><code><a href="/pkg/runtime/debug/#ReadGCStats">debug.ReadGCStats</a></code>
reads statistics about garbage collection.
It is useful to see how much of the resources are spent on GC pauses.
It also reports a timeline of garbage collector pauses and pause time percentiles.</li>
<li><code><a href="/pkg/runtime/debug/#Stack">debug.Stack</a></code>
returns the current stack trace. Stack trace
is useful to see how many goroutines are currently running,
what they are doing, and whether they are blocked or not.</li>
<li><code><a href="/pkg/runtime/debug/#WriteHeapDump">debug.WriteHeapDump</a></code>
suspends the execution of all goroutines
and allows you to dump the heap to a file. A heap dump is a
snapshot of a Go process' memory at a given time. It contains all
allocated objects as well as goroutines, finalizers, and more.</li>
<li><code><a href="/pkg/runtime#NumGoroutine">runtime.NumGoroutine</a></code>
returns the number of current goroutines.
The value can be monitored to see whether enough goroutines are
utilized, or to detect goroutine leaks.</li>
</ul>

<h3 id="execution-tracer">Execution tracer</h3>

<p>Go comes with a runtime execution tracer to capture a wide range
of runtime events. Scheduling, syscall, garbage collections,
heap size, and other events are collected by runtime and available
for visualization by the go tool trace. Execution tracer is a tool
to detect latency and utilization problems. You can examine how well
the CPU is utilized, and when networking or syscalls are a cause of
preemption for the goroutines.</p>

<p>Tracer is useful to:</p>
<ul>
<li>Understand how your goroutines execute.</li>
<li>Understand some of the core runtime events such as GC runs.</li>
<li>Identify poorly parallelized execution.</li>
</ul>

<p>However, it is not great for identifying hot spots such as
analyzing the cause of excessive memory or CPU usage.
Use profiling tools instead first to address them.</p>

<p>
<img width="800" src="https://storage.googleapis.com/golangorg-assets/tracer-lock.png">
</p>

<p>Above, the go tool trace visualization shows the execution started
fine, and then it became serialized. It suggests that there might
be lock contention for a shared resource that creates a bottleneck.</p>

<p>See <a href="https://golang.org/cmd/trace/"><code>go</code> <code>tool</code> <code>trace</code></a>
to collect and analyze runtime traces.
</p>

<h3 id="godebug">GODEBUG</h3>

<p>Runtime also emits events and information if
<a href="https://golang.org/pkg/runtime/#hdr-Environment_Variables">GODEBUG</a>
environmental variable is set accordingly.</p>

<ul>
<li>GODEBUG=gctrace=1 prints garbage collector events at
each collection, summarizing the amount of memory collected
and the length of the pause.</li>
<li>GODEBUG=schedtrace=X prints scheduling events every X milliseconds.</li>
</ul>
