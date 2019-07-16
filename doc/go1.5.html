<!--{
	"Title": "Go 1.5 Release Notes",
	"Path":  "/doc/go1.5",
	"Template": true
}-->


<h2 id="introduction">Introduction to Go 1.5</h2>

<p>
The latest Go release, version 1.5,
is a significant release, including major architectural changes to the implementation.
Despite that, we expect almost all Go programs to continue to compile and run as before,
because the release still maintains the Go 1 <a href="/doc/go1compat.html">promise
of compatibility</a>.
</p>

<p>
The biggest developments in the implementation are:
</p>

<ul>

<li>
The compiler and runtime are now written entirely in Go (with a little assembler).
C is no longer involved in the implementation, and so the C compiler that was
once necessary for building the distribution is gone.
</li>

<li>
The garbage collector is now <a href="https://golang.org/s/go14gc">concurrent</a> and provides dramatically lower
pause times by running, when possible, in parallel with other goroutines.
</li>

<li>
By default, Go programs run with <code>GOMAXPROCS</code> set to the
number of cores available; in prior releases it defaulted to 1.
</li>

<li>
Support for <a href="https://golang.org/s/go14internal">internal packages</a>
is now provided for all repositories, not just the Go core.
</li>

<li>
The <code>go</code> command now provides <a href="https://golang.org/s/go15vendor">experimental
support</a> for "vendoring" external dependencies.
</li>

<li>
A new <code>go tool trace</code> command supports fine-grained
tracing of program execution.
</li>

<li>
A new <code>go doc</code> command (distinct from <code>godoc</code>)
is customized for command-line use.
</li>

</ul>

<p>
These and a number of other changes to the implementation and tools
are discussed below.
</p>

<p>
The release also contains one small language change involving map literals.
</p>

<p>
Finally, the timing of the <a href="https://golang.org/s/releasesched">release</a>
strays from the usual six-month interval,
both to provide more time to prepare this major release and to shift the schedule thereafter to
time the release dates more conveniently.
</p>

<h2 id="language">Changes to the language</h2>

<h3 id="map_literals">Map literals</h3>

<p>
Due to an oversight, the rule that allowed the element type to be elided from slice literals was not
applied to map keys.
This has been <a href="/cl/2591">corrected</a> in Go 1.5.
An example will make this clear.
As of Go 1.5, this map literal,
</p>

<pre>
m := map[Point]string{
    Point{29.935523, 52.891566}:   "Persepolis",
    Point{-25.352594, 131.034361}: "Uluru",
    Point{37.422455, -122.084306}: "Googleplex",
}
</pre>

<p>
may be written as follows, without the <code>Point</code> type listed explicitly:
</p>

<pre>
m := map[Point]string{
    {29.935523, 52.891566}:   "Persepolis",
    {-25.352594, 131.034361}: "Uluru",
    {37.422455, -122.084306}: "Googleplex",
}
</pre>

<h2 id="implementation">The Implementation</h2>

<h3 id="c">No more C</h3>

<p>
The compiler and runtime are now implemented in Go and assembler, without C.
The only C source left in the tree is related to testing or to <code>cgo</code>.
There was a C compiler in the tree in 1.4 and earlier.
It was used to build the runtime; a custom compiler was necessary in part to
guarantee the C code would work with the stack management of goroutines.
Since the runtime is in Go now, there is no need for this C compiler and it is gone.
Details of the process to eliminate C are discussed <a href="https://golang.org/s/go13compiler">elsewhere</a>.
</p>

<p>
The conversion from C was done with the help of custom tools created for the job.
Most important, the compiler was actually moved by automatic translation of
the C code into Go.
It is in effect the same program in a different language.
It is not a new implementation
of the compiler so we expect the process will not have introduced new compiler
bugs.
An overview of this process is available in the slides for
<a href="https://talks.golang.org/2015/gogo.slide">this presentation</a>.
</p>

<h3 id="compiler_and_tools">Compiler and tools</h3>

<p>
Independent of but encouraged by the move to Go, the names of the tools have changed.
The old names <code>6g</code>, <code>8g</code> and so on are gone; instead there
is just one binary, accessible as <code>go</code> <code>tool</code> <code>compile</code>,
that compiles Go source into binaries suitable for the architecture and operating system
specified by <code>$GOARCH</code> and <code>$GOOS</code>.
Similarly, there is now one linker (<code>go</code> <code>tool</code> <code>link</code>)
and one assembler (<code>go</code> <code>tool</code> <code>asm</code>).
The linker was translated automatically from the old C implementation,
but the assembler is a new native Go implementation discussed
in more detail below.
</p>

<p>
Similar to the drop of the names <code>6g</code>, <code>8g</code>, and so on,
the output of the compiler and assembler are now given a plain <code>.o</code> suffix
rather than <code>.8</code>, <code>.6</code>, etc.
</p>


<h3 id="gc">Garbage collector</h3>

<p>
The garbage collector has been re-engineered for 1.5 as part of the development
outlined in the <a href="https://golang.org/s/go14gc">design document</a>.
Expected latencies are much lower than with the collector
in prior releases, through a combination of advanced algorithms,
better <a href="https://golang.org/s/go15gcpacing">scheduling</a> of the collector,
and running more of the collection in parallel with the user program.
The "stop the world" phase of the collector
will almost always be under 10 milliseconds and usually much less.
</p>

<p>
For systems that benefit from low latency, such as user-responsive web sites,
the drop in expected latency with the new collector may be important.
</p>

<p>
Details of the new collector were presented in a
<a href="https://talks.golang.org/2015/go-gc.pdf">talk</a> at GopherCon 2015.
</p>

<h3 id="runtime">Runtime</h3>

<p>
In Go 1.5, the order in which goroutines are scheduled has been changed.
The properties of the scheduler were never defined by the language,
but programs that depend on the scheduling order may be broken
by this change.
We have seen a few (erroneous) programs affected by this change.
If you have programs that implicitly depend on the scheduling
order, you will need to update them.
</p>

<p>
Another potentially breaking change is that the runtime now
sets the default number of threads to run simultaneously,
defined by <code>GOMAXPROCS</code>, to the number
of cores available on the CPU.
In prior releases the default was 1.
Programs that do not expect to run with multiple cores may
break inadvertently.
They can be updated by removing the restriction or by setting
<code>GOMAXPROCS</code> explicitly.
For a more detailed discussion of this change, see
the <a href="https://golang.org/s/go15gomaxprocs">design document</a>.
</p>

<h3 id="build">Build</h3>

<p>
Now that the Go compiler and runtime are implemented in Go, a Go compiler
must be available to compile the distribution from source.
Thus, to build the Go core, a working Go distribution must already be in place.
(Go programmers who do not work on the core are unaffected by this change.)
Any Go 1.4 or later distribution (including <code>gccgo</code>) will serve.
For details, see the <a href="https://golang.org/s/go15bootstrap">design document</a>.
</p>

<h2 id="ports">Ports</h2>

<p>
Due mostly to the industry's move away from the 32-bit x86 architecture,
the set of binary downloads provided is reduced in 1.5.
A distribution for the OS X operating system is provided only for the
<code>amd64</code> architecture, not <code>386</code>.
Similarly, the ports for Snow Leopard (Apple OS X 10.6) still work but are no
longer released as a download or maintained since Apple no longer maintains that version
of the operating system.
Also, the <code>dragonfly/386</code> port is no longer supported at all
because DragonflyBSD itself no longer supports the 32-bit 386 architecture.
</p>

<p>
There are however several new ports available to be built from source.
These include <code>darwin/arm</code> and <code>darwin/arm64</code>.
The new port <code>linux/arm64</code> is mostly in place, but <code>cgo</code>
is only supported using external linking.
</p>

<p>
Also available as experiments are <code>ppc64</code>
and <code>ppc64le</code> (64-bit PowerPC, big- and little-endian).
Both these ports support <code>cgo</code> but
only with internal linking.
</p>

<p>
On FreeBSD, Go 1.5 requires FreeBSD 8-STABLE+ because of its new use of the <code>SYSCALL</code> instruction.
</p>

<p>
On NaCl, Go 1.5 requires SDK version pepper-41. Later pepper versions are not
compatible due to the removal of the sRPC subsystem from the NaCl runtime.
</p>

<p>
On Darwin, the use of the system X.509 certificate interface can be disabled
with the <code>ios</code> build tag.
</p>

<p>
The Solaris port now has full support for cgo and the packages
<a href="/pkg/net/"><code>net</code></a> and
<a href="/pkg/crypto/x509/"><code>crypto/x509</code></a>,
as well as a number of other fixes and improvements.
</p>

<h2 id="tools">Tools</h2>

<h3 id="translate">Translating</h3>

<p>
As part of the process to eliminate C from the tree, the compiler and
linker were translated from C to Go.
It was a genuine (machine assisted) translation, so the new programs are essentially
the old programs translated rather than new ones with new bugs.
We are confident the translation process has introduced few if any new bugs,
and in fact uncovered a number of previously unknown bugs, now fixed.
</p>

<p>
The assembler is a new program, however; it is described below.
</p>

<h3 id="rename">Renaming</h3>

<p>
The suites of programs that were the compilers (<code>6g</code>, <code>8g</code>, etc.),
the assemblers (<code>6a</code>, <code>8a</code>, etc.),
and the linkers (<code>6l</code>, <code>8l</code>, etc.)
have each been consolidated into a single tool that is configured
by the environment variables <code>GOOS</code> and <code>GOARCH</code>.
The old names are gone; the new tools are available through the <code>go</code> <code>tool</code>
mechanism as <code>go tool compile</code>,
<code>go tool asm</code>,
<code>and go tool link</code>.
Also, the file suffixes <code>.6</code>, <code>.8</code>, etc. for the
intermediate object files are also gone; now they are just plain <code>.o</code> files.
</p>

<p>
For example, to build and link a program on amd64 for Darwin
using the tools directly, rather than through <code>go build</code>,
one would run:
</p>

<pre>
$ export GOOS=darwin GOARCH=amd64
$ go tool compile program.go
$ go tool link program.o
</pre>

<h3 id="moving">Moving</h3>

<p>
Because the <a href="/pkg/go/types/"><code>go/types</code></a> package
has now moved into the main repository (see below),
the <a href="/cmd/vet"><code>vet</code></a> and
<a href="/cmd/cover"><code>cover</code></a>
tools have also been moved.
They are no longer maintained in the external <code>golang.org/x/tools</code> repository,
although (deprecated) source still resides there for compatibility with old releases.
</p>

<h3 id="compiler">Compiler</h3>

<p>
As described above, the compiler in Go 1.5 is a single Go program,
translated from the old C source, that replaces <code>6g</code>, <code>8g</code>,
and so on.
Its target is configured by the environment variables <code>GOOS</code> and <code>GOARCH</code>.
</p>

<p>
The 1.5 compiler is mostly equivalent to the old,
but some internal details have changed.
One significant change is that evaluation of constants now uses
the <a href="/pkg/math/big/"><code>math/big</code></a> package
rather than a custom (and less well tested) implementation of high precision
arithmetic.
We do not expect this to affect the results.
</p>

<p>
For the amd64 architecture only, the compiler has a new option, <code>-dynlink</code>,
that assists dynamic linking by supporting references to Go symbols
defined in external shared libraries.
</p>

<h3 id="assembler">Assembler</h3>

<p>
Like the compiler and linker, the assembler in Go 1.5 is a single program
that replaces the suite of assemblers (<code>6a</code>,
<code>8a</code>, etc.) and the environment variables
<code>GOARCH</code> and <code>GOOS</code>
configure the architecture and operating system.
Unlike the other programs, the assembler is a wholly new program
written in Go.
</p>

 <p>
The new assembler is very nearly compatible with the previous
ones, but there are a few changes that may affect some
assembler source files.
See the updated <a href="/doc/asm">assembler guide</a>
for more specific information about these changes. In summary:

</p>

<p>
First, the expression evaluation used for constants is a little
different.
It now uses unsigned 64-bit arithmetic and the precedence
of operators (<code>+</code>, <code>-</code>, <code><<</code>, etc.)
comes from Go, not C.
We expect these changes to affect very few programs but
manual verification may be required.
</p>

<p>
Perhaps more important is that on machines where
<code>SP</code> or <code>PC</code> is only an alias
for a numbered register,
such as <code>R13</code> for the stack pointer and
<code>R15</code> for the hardware program counter
on ARM,
a reference to such a register that does not include a symbol
is now illegal.
For example, <code>SP</code> and <code>4(SP)</code> are
illegal but <code>sym+4(SP)</code> is fine.
On such machines, to refer to the hardware register use its
true <code>R</code> name.
</p>

<p>
One minor change is that some of the old assemblers
permitted the notation
</p>

<pre>
constant=value
</pre>

<p>
to define a named constant.
Since this is always possible to do with the traditional
C-like <code>#define</code> notation, which is still
supported (the assembler includes an implementation
of a simplified C preprocessor), the feature was removed.
</p>

<h3 id="link">Linker</h3>

<p>
The linker in Go 1.5 is now one Go program,
that replaces <code>6l</code>, <code>8l</code>, etc.
Its operating system and instruction set are specified
by the environment variables <code>GOOS</code> and <code>GOARCH</code>.
</p>

<p>
There are several other changes.
The most significant is the addition of a <code>-buildmode</code> option that
expands the style of linking; it now supports
situations such as building shared libraries and allowing other languages
to call into Go libraries.
Some of these were outlined in a <a href="https://golang.org/s/execmodes">design document</a>.
For a list of the available build modes and their use, run
</p>

<pre>
$ go help buildmode
</pre>

<p>
Another minor change is that the linker no longer records build time stamps in
the header of Windows executables.
Also, although this may be fixed, Windows cgo executables are missing some
DWARF information.
</p>

<p>
Finally, the <code>-X</code> flag, which takes two arguments,
as in
</p>

<pre>
-X importpath.name value
</pre>

<p>
now also accepts a more common Go flag style with a single argument
that is itself a <code>name=value</code> pair:
</p>

<pre>
-X importpath.name=value
</pre>

<p>
Although the old syntax still works, it is recommended that uses of this
flag in scripts and the like be updated to the new form.
</p>

<h3 id="go_command">Go command</h3>

<p>
The <a href="/cmd/go"><code>go</code></a> command's basic operation
is unchanged, but there are a number of changes worth noting.
</p>

<p>
The previous release introduced the idea of a directory internal to a package
being unimportable through the <code>go</code> command.
In 1.4, it was tested with the introduction of some internal elements
in the core repository.
As suggested in the <a href="https://golang.org/s/go14internal">design document</a>,
that change is now being made available to all repositories.
The rules are explained in the design document, but in summary any
package in or under a directory named <code>internal</code> may
be imported by packages rooted in the same subtree.
Existing packages with directory elements named <code>internal</code> may be
inadvertently broken by this change, which was why it was advertised
in the last release.
</p>

<p>
Another change in how packages are handled is the experimental
addition of support for "vendoring".
For details, see the documentation for the <a href="/cmd/go/#hdr-Vendor_Directories"><code>go</code> command</a>
and the <a href="https://golang.org/s/go15vendor">design document</a>.
</p>

<p>
There have also been several minor changes.
Read the <a href="/cmd/go">documentation</a> for full details.
</p>

<ul>

<li>
SWIG support has been updated such that
<code>.swig</code> and <code>.swigcxx</code>
now require SWIG 3.0.6 or later.
</li>

<li>
The <code>install</code> subcommand now removes the
binary created by the <code>build</code> subcommand
in the source directory, if present,
to avoid problems having two binaries present in the tree.
</li>

<li>
The <code>std</code> (standard library) wildcard package name
now excludes commands.
A new <code>cmd</code> wildcard covers the commands.
</li>

<li>
A new <code>-asmflags</code> build option
sets flags to pass to the assembler.
However,
the <code>-ccflags</code> build option has been dropped;
it was specific to the old, now deleted C compiler .
</li>

<li>
A new <code>-buildmode</code> build option
sets the build mode, described above.
</li>

<li>
A new <code>-pkgdir</code> build option
sets the location of installed package archives,
to help isolate custom builds.
</li>

<li>
A new <code>-toolexec</code> build option
allows substitution of a different command to invoke
the compiler and so on.
This acts as a custom replacement for <code>go tool</code>.
</li>

<li>
The <code>test</code> subcommand now has a <code>-count</code>
flag to specify how many times to run each test and benchmark.
The <a href="/pkg/testing/"><code>testing</code></a> package
does the work here, through the <code>-test.count</code> flag.
</li>

<li>
The <code>generate</code> subcommand has a couple of new features.
The <code>-run</code> option specifies a regular expression to select which directives
to execute; this was proposed but never implemented in 1.4.
The executing pattern now has access to two new environment variables:
<code>$GOLINE</code> returns the source line number of the directive
and <code>$DOLLAR</code> expands to a dollar sign.
</li>

<li>
The <code>get</code> subcommand now has a <code>-insecure</code>
flag that must be enabled if fetching from an insecure repository, one that
does not encrypt the connection.
</li>

</ul>

<h3 id="vet_command">Go vet command</h3>

<p>
The <a href="/cmd/vet"><code>go tool vet</code></a> command now does
more thorough validation of struct tags.
</p>

<h3 id="trace_command">Trace command</h3>

<p>
A new tool is available for dynamic execution tracing of Go programs.
The usage is analogous to how the test coverage tool works.
Generation of traces is integrated into <code>go test</code>,
and then a separate execution of the tracing tool itself analyzes the results:
</p>

<pre>
$ go test -trace=trace.out path/to/package
$ go tool trace [flags] pkg.test trace.out
</pre>

<p>
The flags enable the output to be displayed in a browser window.
For details, run <code>go tool trace -help</code>.
There is also a description of the tracing facility in this
<a href="https://talks.golang.org/2015/dynamic-tools.slide">talk</a>
from GopherCon 2015.
</p>

<h3 id="doc_command">Go doc command</h3>

<p>
A few releases back, the <code>go doc</code>
command was deleted as being unnecessary.
One could always run "<code>godoc .</code>" instead.
The 1.5 release introduces a new <a href="/cmd/doc"><code>go doc</code></a>
command with a more convenient command-line interface than
<code>godoc</code>'s.
It is designed for command-line usage specifically, and provides a more
compact and focused presentation of the documentation for a package
or its elements, according to the invocation.
It also provides case-insensitive matching and
support for showing the documentation for unexported symbols.
For details run "<code>go help doc</code>".
</p>

<h3 id="cgo">Cgo</h3>

<p>
When parsing <code>#cgo</code> lines,
the invocation <code>${SRCDIR}</code> is now
expanded into the path to the source directory.
This allows options to be passed to the
compiler and linker that involve file paths relative to the
source code directory. Without the expansion the paths would be
invalid when the current working directory changes.
</p>

<p>
Solaris now has full cgo support.
</p>

<p>
On Windows, cgo now uses external linking by default.
</p>

<p>
When a C struct ends with a zero-sized field, but the struct itself is
not zero-sized, Go code can no longer refer to the zero-sized field.
Any such references will have to be rewritten.
</p>

<h2 id="performance">Performance</h2>

<p>
As always, the changes are so general and varied that precise statements
about performance are difficult to make.
The changes are even broader ranging than usual in this release, which
includes a new garbage collector and a conversion of the runtime to Go.
Some programs may run faster, some slower.
On average the programs in the Go 1 benchmark suite run a few percent faster in Go 1.5
than they did in Go 1.4,
while as mentioned above the garbage collector's pauses are
dramatically shorter, and almost always under 10 milliseconds.
</p>

<p>
Builds in Go 1.5 will be slower by a factor of about two.
The automatic translation of the compiler and linker from C to Go resulted in
unidiomatic Go code that performs poorly compared to well-written Go.
Analysis tools and refactoring helped to improve the code, but much remains to be done.
Further profiling and optimization will continue in Go 1.6 and future releases.
For more details, see these <a href="https://talks.golang.org/2015/gogo.slide">slides</a>
and associated <a href="https://www.youtube.com/watch?v=cF1zJYkBW4A">video</a>.
</p>

<h2 id="library">Core library</h2>

<h3 id="flag">Flag</h3>

<p>
The flag package's
<a href="/pkg/flag/#PrintDefaults"><code>PrintDefaults</code></a>
function, and method on <a href="/pkg/flag/#FlagSet"><code>FlagSet</code></a>,
have been modified to create nicer usage messages.
The format has been changed to be more human-friendly and in the usage
messages a word quoted with `backquotes` is taken to be the name of the
flag's operand to display in the usage message.
For instance, a flag created with the invocation,
</p>

<pre>
cpuFlag = flag.Int("cpu", 1, "run `N` processes in parallel")
</pre>

<p>
will show the help message,
</p>

<pre>
-cpu N
    	run N processes in parallel (default 1)
</pre>

<p>
Also, the default is now listed only when it is not the zero value for the type.
</p>

<h3 id="math_big">Floats in math/big</h3>

<p>
The <a href="/pkg/math/big/"><code>math/big</code></a> package
has a new, fundamental data type,
<a href="/pkg/math/big/#Float"><code>Float</code></a>,
which implements arbitrary-precision floating-point numbers.
A <code>Float</code> value is represented by a boolean sign,
a variable-length mantissa, and a 32-bit fixed-size signed exponent.
The precision of a <code>Float</code> (the mantissa size in bits)
can be specified explicitly or is otherwise determined by the first
operation that creates the value.
Once created, the size of a <code>Float</code>'s mantissa may be modified with the
<a href="/pkg/math/big/#Float.SetPrec"><code>SetPrec</code></a> method.
<code>Floats</code> support the concept of infinities, such as are created by
overflow, but values that would lead to the equivalent of IEEE 754 NaNs
trigger a panic.
<code>Float</code> operations support all IEEE-754 rounding modes.
When the precision is set to 24 (53) bits,
operations that stay within the range of normalized <code>float32</code>
(<code>float64</code>)
values produce the same results as the corresponding IEEE-754
arithmetic on those values.
</p>

<h3 id="go_types">Go types</h3>

<p>
The <a href="/pkg/go/types/"><code>go/types</code></a> package
up to now has been maintained in the <code>golang.org/x</code>
repository; as of Go 1.5 it has been relocated to the main repository.
The code at the old location is now deprecated.
There is also a modest API change in the package, discussed below.
</p>

<p>
Associated with this move, the
<a href="/pkg/go/constant/"><code>go/constant</code></a>
package also moved to the main repository;
it was <code>golang.org/x/tools/exact</code> before.
The <a href="/pkg/go/importer/"><code>go/importer</code></a> package
also moved to the main repository,
as well as some tools described above.
</p>

<h3 id="net">Net</h3>

<p>
The DNS resolver in the net package has almost always used <code>cgo</code> to access
the system interface.
A change in Go 1.5 means that on most Unix systems DNS resolution
will no longer require <code>cgo</code>, which simplifies execution
on those platforms.
Now, if the system's networking configuration permits, the native Go resolver
will suffice.
The important effect of this change is that each DNS resolution occupies a goroutine
rather than a thread,
so a program with multiple outstanding DNS requests will consume fewer operating
system resources.
</p>

<p>
The decision of how to run the resolver applies at run time, not build time.
The <code>netgo</code> build tag that has been used to enforce the use
of the Go resolver is no longer necessary, although it still works.
A new <code>netcgo</code> build tag forces the use of the <code>cgo</code> resolver at
build time.
To force <code>cgo</code> resolution at run time set
<code>GODEBUG=netdns=cgo</code> in the environment.
More debug options are documented <a href="https://golang.org/cl/11584">here</a>.
</p>

<p>
This change applies to Unix systems only.
Windows, Mac OS X, and Plan 9 systems behave as before.
</p>

<h3 id="reflect">Reflect</h3>

<p>
The <a href="/pkg/reflect/"><code>reflect</code></a> package
has two new functions: <a href="/pkg/reflect/#ArrayOf"><code>ArrayOf</code></a>
and <a href="/pkg/reflect/#FuncOf"><code>FuncOf</code></a>.
These functions, analogous to the extant
<a href="/pkg/reflect/#SliceOf"><code>SliceOf</code></a> function,
create new types at runtime to describe arrays and functions.
</p>

<h3 id="hardening">Hardening</h3>

<p>
Several dozen bugs were found in the standard library
through randomized testing with the
<a href="https://github.com/dvyukov/go-fuzz"><code>go-fuzz</code></a> tool.
Bugs were fixed in the
<a href="/pkg/archive/tar/"><code>archive/tar</code></a>,
<a href="/pkg/archive/zip/"><code>archive/zip</code></a>,
<a href="/pkg/compress/flate/"><code>compress/flate</code></a>,
<a href="/pkg/encoding/gob/"><code>encoding/gob</code></a>,
<a href="/pkg/fmt/"><code>fmt</code></a>,
<a href="/pkg/html/template/"><code>html/template</code></a>,
<a href="/pkg/image/gif/"><code>image/gif</code></a>,
<a href="/pkg/image/jpeg/"><code>image/jpeg</code></a>,
<a href="/pkg/image/png/"><code>image/png</code></a>, and
<a href="/pkg/text/template/"><code>text/template</code></a>,
packages.
The fixes harden the implementation against incorrect and malicious inputs.
</p>

<h3 id="minor_library_changes">Minor changes to the library</h3>

<ul>

<li>
The <a href="/pkg/archive/zip/"><code>archive/zip</code></a> package's
<a href="/pkg/archive/zip/#Writer"><code>Writer</code></a> type now has a
<a href="/pkg/archive/zip/#Writer.SetOffset"><code>SetOffset</code></a>
method to specify the location within the output stream at which to write the archive.
</li>

<li>
The <a href="/pkg/bufio/#Reader"><code>Reader</code></a> in the
<a href="/pkg/bufio/"><code>bufio</code></a> package now has a
<a href="/pkg/bufio/#Reader.Discard"><code>Discard</code></a>
method to discard data from the input.
</li>

<li>
In the <a href="/pkg/bytes/"><code>bytes</code></a> package,
the <a href="/pkg/bytes/#Buffer"><code>Buffer</code></a> type
now has a <a href="/pkg/bytes/#Buffer.Cap"><code>Cap</code></a> method
that reports the number of bytes allocated within the buffer.
Similarly, in both the <a href="/pkg/bytes/"><code>bytes</code></a>
and <a href="/pkg/strings/"><code>strings</code></a> packages,
the <a href="/pkg/bytes/#Reader"><code>Reader</code></a>
type now has a <a href="/pkg/bytes/#Reader.Size"><code>Size</code></a>
method that reports the original length of the underlying slice or string.
</li>

<li>
Both the <a href="/pkg/bytes/"><code>bytes</code></a> and
<a href="/pkg/strings/"><code>strings</code></a> packages
also now have a <a href="/pkg/bytes/#LastIndexByte"><code>LastIndexByte</code></a>
function that locates the rightmost byte with that value in the argument.
</li>

<li>
The <a href="/pkg/crypto/"><code>crypto</code></a> package
has a new interface, <a href="/pkg/crypto/#Decrypter"><code>Decrypter</code></a>,
that abstracts the behavior of a private key used in asymmetric decryption.
</li>

<li>
In the <a href="/pkg/crypto/cipher/"><code>crypto/cipher</code></a> package,
the documentation for the <a href="/pkg/crypto/cipher/#Stream"><code>Stream</code></a>
interface has been clarified regarding the behavior when the source and destination are
different lengths.
If the destination is shorter than the source, the method will panic.
This is not a change in the implementation, only the documentation.
</li>

<li>
Also in the <a href="/pkg/crypto/cipher/"><code>crypto/cipher</code></a> package,
there is now support for nonce lengths other than 96 bytes in AES's Galois/Counter mode (GCM),
which some protocols require.
</li>

<li>
In the <a href="/pkg/crypto/elliptic/"><code>crypto/elliptic</code></a> package,
there is now a <code>Name</code> field in the
<a href="/pkg/crypto/elliptic/#CurveParams"><code>CurveParams</code></a> struct,
and the curves implemented in the package have been given names.
These names provide a safer way to select a curve, as opposed to
selecting its bit size, for cryptographic systems that are curve-dependent.
</li>

<li>
Also in the <a href="/pkg/crypto/elliptic/"><code>crypto/elliptic</code></a> package,
the <a href="/pkg/crypto/elliptic/#Unmarshal"><code>Unmarshal</code></a> function
now verifies that the point is actually on the curve.
(If it is not, the function returns nils).
This change guards against certain attacks.
</li>

<li>
The <a href="/pkg/crypto/sha512/"><code>crypto/sha512</code></a>
package now has support for the two truncated versions of
the SHA-512 hash algorithm, SHA-512/224 and SHA-512/256.
</li>

<li>
The <a href="/pkg/crypto/tls/"><code>crypto/tls</code></a> package
minimum protocol version now defaults to TLS 1.0.
The old default, SSLv3, is still available through <a href="/pkg/crypto/tls/#Config"><code>Config</code></a> if needed.
</li>

<li>
The <a href="/pkg/crypto/tls/"><code>crypto/tls</code></a> package
now supports Signed Certificate Timestamps (SCTs) as specified in RFC 6962.
The server serves them if they are listed in the
<a href="/pkg/crypto/tls/#Certificate"><code>Certificate</code></a> struct,
and the client requests them and exposes them, if present,
in its <a href="/pkg/crypto/tls/#ConnectionState"><code>ConnectionState</code></a> struct.

<li>
The stapled OCSP response to a <a href="/pkg/crypto/tls/"><code>crypto/tls</code></a> client connection,
previously only available via the
<a href="/pkg/crypto/tls/#Conn.OCSPResponse"><code>OCSPResponse</code></a> method,
is now exposed in the <a href="/pkg/crypto/tls/#ConnectionState"><code>ConnectionState</code></a> struct.
</li>

<li>
The <a href="/pkg/crypto/tls/"><code>crypto/tls</code></a> server implementation
will now always call the
<code>GetCertificate</code> function in
the <a href="/pkg/crypto/tls/#Config"><code>Config</code></a> struct
to select a certificate for the connection when none is supplied.
</li>

<li>
Finally, the session ticket keys in the
<a href="/pkg/crypto/tls/"><code>crypto/tls</code></a> package
can now be changed while the server is running.
This is done through the new
<a href="/pkg/crypto/tls/#Config.SetSessionTicketKeys"><code>SetSessionTicketKeys</code></a>
method of the
<a href="/pkg/crypto/tls/#Config"><code>Config</code></a> type.
</li>

<li>
In the <a href="/pkg/crypto/x509/"><code>crypto/x509</code></a> package,
wildcards are now accepted only in the leftmost label as defined in
<a href="https://tools.ietf.org/html/rfc6125#section-6.4.3">the specification</a>.
</li>

<li>
Also in the <a href="/pkg/crypto/x509/"><code>crypto/x509</code></a> package,
the handling of unknown critical extensions has been changed.
They used to cause parse errors but now they are parsed and caused errors only
in <a href="/pkg/crypto/x509/#Certificate.Verify"><code>Verify</code></a>.
The new field <code>UnhandledCriticalExtensions</code> of
<a href="/pkg/crypto/x509/#Certificate"><code>Certificate</code></a> records these extensions.
</li>

<li>
The <a href="/pkg/database/sql/#DB"><code>DB</code></a> type of the
<a href="/pkg/database/sql/"><code>database/sql</code></a> package
now has a <a href="/pkg/database/sql/#DB.Stats"><code>Stats</code></a> method
to retrieve database statistics.
</li>

<li>
The <a href="/pkg/debug/dwarf/"><code>debug/dwarf</code></a>
package has extensive additions to better support DWARF version 4.
See for example the definition of the new type
<a href="/pkg/debug/dwarf/#Class"><code>Class</code></a>.
</li>

<li>
The <a href="/pkg/debug/dwarf/"><code>debug/dwarf</code></a> package
also now supports decoding of DWARF line tables.
</li>

<li>
The <a href="/pkg/debug/elf/"><code>debug/elf</code></a>
package now has support for the 64-bit PowerPC architecture.
</li>

<li>
The <a href="/pkg/encoding/base64/"><code>encoding/base64</code></a> package
now supports unpadded encodings through two new encoding variables,
<a href="/pkg/encoding/base64/#RawStdEncoding"><code>RawStdEncoding</code></a> and
<a href="/pkg/encoding/base64/#RawURLEncoding"><code>RawURLEncoding</code></a>.
</li>

<li>
The <a href="/pkg/encoding/json/"><code>encoding/json</code></a> package
now returns an <a href="/pkg/encoding/json/#UnmarshalTypeError"><code>UnmarshalTypeError</code></a>
if a JSON value is not appropriate for the target variable or component
to which it is being unmarshaled.
</li>

<li>
The <code>encoding/json</code>'s
<a href="/pkg/encoding/json/#Decoder"><code>Decoder</code></a>
type has a new method that provides a streaming interface for decoding
a JSON document:
<a href="/pkg/encoding/json/#Decoder.Token"><code>Token</code></a>.
It also interoperates with the existing functionality of <code>Decode</code>,
which will continue a decode operation already started with <code>Decoder.Token</code>.
</li>

<li>
The <a href="/pkg/flag/"><code>flag</code></a> package
has a new function, <a href="/pkg/flag/#UnquoteUsage"><code>UnquoteUsage</code></a>,
to assist in the creation of usage messages using the new convention
described above.
</li>

<li>
In the <a href="/pkg/fmt/"><code>fmt</code></a> package,
a value of type <a href="/pkg/reflect/#Value"><code>Value</code></a> now
prints what it holds, rather than use the <code>reflect.Value</code>'s <code>Stringer</code>
method, which produces things like <code>&lt;int Value&gt;</code>.
</li>

<li>
The <a href="/pkg/ast/#EmptyStmt"><code>EmptyStmt</code></a> type
in the <a href="/pkg/go/ast/"><code>go/ast</code></a> package now
has a boolean <code>Implicit</code> field that records whether the
semicolon was implicitly added or was present in the source.
</li>

<li>
For forward compatibility the <a href="/pkg/go/build/"><code>go/build</code></a> package
reserves <code>GOARCH</code> values for  a number of architectures that Go might support one day.
This is not a promise that it will.
Also, the <a href="/pkg/go/build/#Package"><code>Package</code></a> struct
now has a <code>PkgTargetRoot</code> field that stores the
architecture-dependent root directory in which to install, if known.
</li>

<li>
The (newly migrated) <a href="/pkg/go/types/"><code>go/types</code></a>
package allows one to control the prefix attached to package-level names using
the new <a href="/pkg/go/types/#Qualifier"><code>Qualifier</code></a>
function type as an argument to several functions. This is an API change for
the package, but since it is new to the core, it is not breaking the Go 1 compatibility
rules since code that uses the package must explicitly ask for it at its new location.
To update, run
<a href="https://golang.org/cmd/go/#hdr-Run_go_tool_fix_on_packages"><code>go fix</code></a> on your package.
</li>

<li>
In the <a href="/pkg/image/"><code>image</code></a> package,
the <a href="/pkg/image/#Rectangle"><code>Rectangle</code></a> type
now implements the <a href="/pkg/image/#Image"><code>Image</code></a> interface,
so a <code>Rectangle</code> can serve as a mask when drawing.
</li>

<li>
Also in the <a href="/pkg/image/"><code>image</code></a> package,
to assist in the handling of some JPEG images,
there is now support for 4:1:1 and 4:1:0 YCbCr subsampling and basic
CMYK support, represented by the new <code>image.CMYK</code> struct.
</li>

<li>
The <a href="/pkg/image/color/"><code>image/color</code></a> package
adds basic CMYK support, through the new
<a href="/pkg/image/color/#CMYK"><code>CMYK</code></a> struct,
the <a href="/pkg/image/color/#CMYKModel"><code>CMYKModel</code></a> color model, and the
<a href="/pkg/image/color/#CMYKToRGB"><code>CMYKToRGB</code></a> function, as
needed by some JPEG images.
</li>

<li>
Also in the <a href="/pkg/image/color/"><code>image/color</code></a> package,
the conversion of a <a href="/pkg/image/color/#YCbCr"><code>YCbCr</code></a>
value to <code>RGBA</code> has become more precise.
Previously, the low 8 bits were just an echo of the high 8 bits;
now they contain more accurate information.
Because of the echo property of the old code, the operation
<code>uint8(r)</code> to extract an 8-bit red value worked, but is incorrect.
In Go 1.5, that operation may yield a different value.
The correct code is, and always was, to select the high 8 bits:
<code>uint8(r&gt;&gt;8)</code>.
Incidentally, the <code>image/draw</code> package
provides better support for such conversions; see
<a href="https://blog.golang.org/go-imagedraw-package">this blog post</a>
for more information.
</li>

<li>
Finally, as of Go 1.5 the closest match check in
<a href="/pkg/image/color/#Palette.Index"><code>Index</code></a>
now honors the alpha channel.
</li>

<li>
The <a href="/pkg/image/gif/"><code>image/gif</code></a> package
includes a couple of generalizations.
A multiple-frame GIF file can now have an overall bounds different
from all the contained single frames' bounds.
Also, the <a href="/pkg/image/gif/#GIF"><code>GIF</code></a> struct
now has a <code>Disposal</code> field
that specifies the disposal method for each frame.
</li>

<li>
The <a href="/pkg/io/"><code>io</code></a> package
adds a <a href="/pkg/io/#CopyBuffer"><code>CopyBuffer</code></a> function
that is like <a href="/pkg/io/#Copy"><code>Copy</code></a> but
uses a caller-provided buffer, permitting control of allocation and buffer size.
</li>

<li>
The <a href="/pkg/log/"><code>log</code></a> package
has a new <a href="/pkg/log/#LUTC"><code>LUTC</code></a> flag
that causes time stamps to be printed in the UTC time zone.
It also adds a <a href="/pkg/log/#Logger.SetOutput"><code>SetOutput</code></a> method
for user-created loggers.
</li>

<li>
In Go 1.4, <a href="/pkg/math/#Max"><code>Max</code></a> was not detecting all possible NaN bit patterns.
This is fixed in Go 1.5, so programs that use <code>math.Max</code> on data including NaNs may behave differently,
but now correctly according to the IEEE754 definition of NaNs.
</li>

<li>
The <a href="/pkg/math/big/"><code>math/big</code></a> package
adds a new <a href="/pkg/math/big/#Jacobi"><code>Jacobi</code></a>
function for integers and a new
<a href="/pkg/math/big/#Int.ModSqrt"><code>ModSqrt</code></a>
method for the <a href="/pkg/math/big/#Int"><code>Int</code></a> type.
</li>

<li>
The mime package
adds a new <a href="/pkg/mime/#WordDecoder"><code>WordDecoder</code></a> type
to decode MIME headers containing RFC 204-encoded words.
It also provides <a href="/pkg/mime/#BEncoding"><code>BEncoding</code></a> and
<a href="/pkg/mime/#QEncoding"><code>QEncoding</code></a>
as implementations of the encoding schemes of RFC 2045 and RFC 2047.
</li>

<li>
The <a href="/pkg/mime/"><code>mime</code></a> package also adds an
<a href="/pkg/mime/#ExtensionsByType"><code>ExtensionsByType</code></a>
function that returns the MIME extensions know to be associated with a given MIME type.
</li>

<li>
There is a new <a href="/pkg/mime/quotedprintable/"><code>mime/quotedprintable</code></a>
package that implements the quoted-printable encoding defined by RFC 2045.
</li>

<li>
The <a href="/pkg/net/"><code>net</code></a> package will now
<a href="/pkg/net/#Dial"><code>Dial</code></a> hostnames by trying each
IP address in order until one succeeds.
The <code><a href="/pkg/net/#Dialer">Dialer</a>.DualStack</code>
mode now implements Happy Eyeballs
(<a href="https://tools.ietf.org/html/rfc6555">RFC 6555</a>) by giving the
first address family a 300ms head start; this value can be overridden by
the new <code>Dialer.FallbackDelay</code>.
</li>

<li>
A number of inconsistencies in the types returned by errors in the
<a href="/pkg/net/"><code>net</code></a> package have been
tidied up.
Most now return an
<a href="/pkg/net/#OpError"><code>OpError</code></a> value
with more information than before.
Also, the <a href="/pkg/net/#OpError"><code>OpError</code></a>
type now includes a <code>Source</code> field that holds the local
network address.
</li>

<li>
The <a href="/pkg/net/http/"><code>net/http</code></a> package now
has support for setting trailers from a server <a href="/pkg/net/http/#Handler"><code>Handler</code></a>.
For details, see the documentation for
<a href="/pkg/net/http/#ResponseWriter"><code>ResponseWriter</code></a>.
</li>

<li>
There is a new method to cancel a <a href="/pkg/net/http/"><code>net/http</code></a>
<code>Request</code> by setting the new
<a href="/pkg/net/http/#Request"><code>Request.Cancel</code></a>
field.
It is supported by <code>http.Transport</code>.
The <code>Cancel</code> field's type is compatible with the
<a href="https://godoc.org/golang.org/x/net/context"><code>context.Context.Done</code></a>
return value.
</li>

<li>
Also in the <a href="/pkg/net/http/"><code>net/http</code></a> package,
there is code to ignore the zero <a href="/pkg/time/#Time"><code>Time</code></a> value
in the <a href="/pkg/net/#ServeContent"><code>ServeContent</code></a> function.
As of Go 1.5, it now also ignores a time value equal to the Unix epoch.
</li>

<li>
The <a href="/pkg/net/http/fcgi/"><code>net/http/fcgi</code></a> package
exports two new errors,
<a href="/pkg/net/http/fcgi/#ErrConnClosed"><code>ErrConnClosed</code></a> and
<a href="/pkg/net/http/fcgi/#ErrRequestAborted"><code>ErrRequestAborted</code></a>,
to report the corresponding error conditions.
</li>

<li>
The <a href="/pkg/net/http/cgi/"><code>net/http/cgi</code></a> package
had a bug that mishandled the values of the environment variables
<code>REMOTE_ADDR</code> and <code>REMOTE_HOST</code>.
This has been fixed.
Also, starting with Go 1.5 the package sets the <code>REMOTE_PORT</code>
variable.
</li>

<li>
The <a href="/pkg/net/mail/"><code>net/mail</code></a> package
adds an <a href="/pkg/net/mail/#AddressParser"><code>AddressParser</code></a>
type that can parse mail addresses.
</li>

<li>
The <a href="/pkg/net/smtp/"><code>net/smtp</code></a> package
now has a <a href="/pkg/net/smtp/#Client.TLSConnectionState"><code>TLSConnectionState</code></a>
accessor to the <a href="/pkg/net/smtp/#Client"><code>Client</code></a>
type that returns the client's TLS state.
</li>

<li>
The <a href="/pkg/os/"><code>os</code></a> package
has a new <a href="/pkg/os/#LookupEnv"><code>LookupEnv</code></a> function
that is similar to <a href="/pkg/os/#Getenv"><code>Getenv</code></a>
but can distinguish between an empty environment variable and a missing one.
</li>

<li>
The <a href="/pkg/os/signal/"><code>os/signal</code></a> package
adds new <a href="/pkg/os/signal/#Ignore"><code>Ignore</code></a> and
<a href="/pkg/os/signal/#Reset"><code>Reset</code></a> functions.
</li>

<li>
The <a href="/pkg/runtime/"><code>runtime</code></a>,
<a href="/pkg/runtime/trace/"><code>runtime/trace</code></a>,
and <a href="/pkg/net/http/pprof/"><code>net/http/pprof</code></a> packages
each have new functions to support the tracing facilities described above:
<a href="/pkg/runtime/#ReadTrace"><code>ReadTrace</code></a>,
<a href="/pkg/runtime/#StartTrace"><code>StartTrace</code></a>,
<a href="/pkg/runtime/#StopTrace"><code>StopTrace</code></a>,
<a href="/pkg/runtime/trace/#Start"><code>Start</code></a>,
<a href="/pkg/runtime/trace/#Stop"><code>Stop</code></a>, and
<a href="/pkg/net/http/pprof/#Trace"><code>Trace</code></a>.
See the respective documentation for details.
</li>

<li>
The <a href="/pkg/runtime/pprof/"><code>runtime/pprof</code></a> package
by default now includes overall memory statistics in all memory profiles.
</li>

<li>
The <a href="/pkg/strings/"><code>strings</code></a> package
has a new <a href="/pkg/strings/#Compare"><code>Compare</code></a> function.
This is present to provide symmetry with the <a href="/pkg/bytes/"><code>bytes</code></a> package
but is otherwise unnecessary as strings support comparison natively.
</li>

<li>
The <a href="/pkg/sync/#WaitGroup"><code>WaitGroup</code></a> implementation in
package <a href="/pkg/sync/"><code>sync</code></a>
now diagnoses code that races a call to <a href="/pkg/sync/#WaitGroup.Add"><code>Add</code></a>
against a return from <a href="/pkg/sync/#WaitGroup.Wait"><code>Wait</code></a>.
If it detects this condition, the implementation panics.
</li>

<li>
In the <a href="/pkg/syscall/"><code>syscall</code></a> package,
the Linux <code>SysProcAttr</code> struct now has a
<code>GidMappingsEnableSetgroups</code> field, made necessary
by security changes in Linux 3.19.
On all Unix systems, the struct also has new <code>Foreground</code> and <code>Pgid</code> fields
to provide more control when exec'ing.
On Darwin, there is now a <code>Syscall9</code> function
to support calls with too many arguments.
</li>

<li>
The <a href="/pkg/testing/quick/"><code>testing/quick</code></a> will now
generate <code>nil</code> values for pointer types,
making it possible to use with recursive data structures.
Also, the package now supports generation of array types.
</li>

<li>
In the <a href="/pkg/text/template/"><code>text/template</code></a> and
<a href="/pkg/html/template/"><code>html/template</code></a> packages,
integer constants too large to be represented as a Go integer now trigger a
parse error. Before, they were silently converted to floating point, losing
precision.
</li>

<li>
Also in the <a href="/pkg/text/template/"><code>text/template</code></a> and
<a href="/pkg/html/template/"><code>html/template</code></a> packages,
a new <a href="/pkg/text/template/#Template.Option"><code>Option</code></a> method
allows customization of the behavior of the template during execution.
The sole implemented option allows control over how a missing key is
handled when indexing a map.
The default, which can now be overridden, is as before: to continue with an invalid value.
</li>

<li>
The <a href="/pkg/time/"><code>time</code></a> package's
<code>Time</code> type has a new method
<a href="/pkg/time/#Time.AppendFormat"><code>AppendFormat</code></a>,
which can be used to avoid allocation when printing a time value.
</li>

<li>
The <a href="/pkg/unicode/"><code>unicode</code></a> package and associated
support throughout the system has been upgraded from version 7.0 to
<a href="http://www.unicode.org/versions/Unicode8.0.0/">Unicode 8.0</a>.
</li>

</ul>
