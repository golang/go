<!--{
	"Title": "Go 1.9 Release Notes",
	"Path":  "/doc/go1.9",
	"Template": true
}-->

<!--
NOTE: In this document and others in this directory, the convention is to
set fixed-width phrases with non-fixed-width spaces, as in
<code>hello</code> <code>world</code>.
Do not send CLs removing the interior tags from such phrases.
-->

<style>
ul li { margin: 0.5em 0; }
</style>

<h2 id="introduction">Introduction to Go 1.9</h2>

<p>
  The latest Go release, version 1.9, arrives six months
  after <a href="go1.8">Go 1.8</a> and is the tenth release in
  the <a href="https://golang.org/doc/devel/release.html">Go 1.x
  series</a>.
  There are two <a href="#language">changes to the language</a>:
  adding support for type aliases and defining when implementations
  may fuse floating point operations.
  Most of the changes are in the implementation of the toolchain,
  runtime, and libraries.
  As always, the release maintains the Go 1
  <a href="/doc/go1compat.html">promise of compatibility</a>.
  We expect almost all Go programs to continue to compile and run as
  before.
</p>

<p>
  The release
  adds <a href="#monotonic-time">transparent monotonic time support</a>,
  <a href="#parallel-compile">parallelizes compilation of functions</a> within a package,
  better supports <a href="#test-helper">test helper functions</a>,
  includes a new <a href="#math-bits">bit manipulation package</a>,
  and has a new <a href="#sync-map">concurrent map type</a>.
</p>

<h2 id="language">Changes to the language</h2>

<p>
  There are two changes to the language.
</p>
<p>
  Go now supports type aliases to support gradual code repair while
  moving a type between packages.
  The <a href="https://golang.org/design/18130-type-alias">type alias
  design document</a>
  and <a href="https://talks.golang.org/2016/refactor.article">an
  article on refactoring</a> cover the problem in detail.
  In short, a type alias declaration has the form:
</p>

<pre>
type T1 = T2
</pre>

<p>
  This declaration introduces an alias name <code>T1</code>—an
  alternate spelling—for the type denoted by <code>T2</code>; that is,
  both <code>T1</code> and <code>T2</code> denote the same type.
</p>

<p> <!-- CL 40391 -->
  A smaller language change is that the
  <a href="/ref/spec#Floating_point_operators">language specification
  now states</a> when implementations are allowed to fuse floating
  point operations together, such as by using an architecture's "fused
  multiply and add" (FMA) instruction to compute <code>x*y</code>&nbsp;<code>+</code>&nbsp;<code>z</code>
  without rounding the intermediate result <code>x*y</code>.
  To force the intermediate rounding, write <code>float64(x*y)</code>&nbsp;<code>+</code>&nbsp;<code>z</code>.
</p>

<h2 id="ports">Ports</h2>

<p>
  There are no new supported operating systems or processor
  architectures in this release.
</p>

<h3 id="power8">ppc64x requires POWER8</h3>

<p> <!-- CL 36725, CL 36832 -->
  Both <code>GOARCH=ppc64</code> and <code>GOARCH=ppc64le</code> now
  require at least POWER8 support. In previous releases,
  only <code>GOARCH=ppc64le</code> required POWER8 and the big
  endian <code>ppc64</code> architecture supported older
  hardware.
<p>

<h3 id="freebsd">FreeBSD</h3>

<p>
  Go 1.9 is the last release that will run on FreeBSD 9.3,
  which is already
  <a href="https://www.freebsd.org/security/unsupported.html">unsupported by FreeBSD</a>.
  Go 1.10 will require FreeBSD 10.3+.
</p>

<h3 id="openbsd">OpenBSD 6.0</h3>

<p> <!-- CL 40331 -->
  Go 1.9 now enables PT_TLS generation for cgo binaries and thus
  requires OpenBSD 6.0 or newer. Go 1.9 no longer supports
  OpenBSD 5.9.
<p>

<h3 id="known_issues">Known Issues</h3>

<p>
  There are some instabilities on FreeBSD that are known but not understood.
  These can lead to program crashes in rare cases.
  See <a href="https://golang.org/issue/15658">issue 15658</a>.
  Any help in solving this FreeBSD-specific issue would be appreciated.
</p>

<p>
  Go stopped running NetBSD builders during the Go 1.9 development
  cycle due to NetBSD kernel crashes, up to and including NetBSD 7.1.
  As Go 1.9 is being released, NetBSD 7.1.1 is being released with a fix.
  However, at this time we have no NetBSD builders passing our test suite.
  Any help investigating the
  <a href="https://github.com/golang/go/labels/OS-NetBSD">various NetBSD issues</a>
  would be appreciated.
</p>

<h2 id="tools">Tools</h2>

<h3 id="parallel-compile">Parallel Compilation</h3>

<p>
  The Go compiler now supports compiling a package's functions in parallel, taking
  advantage of multiple cores. This is in addition to the <code>go</code> command's
  existing support for parallel compilation of separate packages.
  Parallel compilation is on by default, but it can be disabled by setting the
  environment variable <code>GO19CONCURRENTCOMPILATION</code> to <code>0</code>.
</p>

<h3 id="vendor-dotdotdot">Vendor matching with ./...</h3>

<p><!-- CL 38745 -->
  By popular request, <code>./...</code> no longer matches packages
  in <code>vendor</code> directories in tools accepting package names,
  such as <code>go</code> <code>test</code>. To match vendor
  directories, write <code>./vendor/...</code>.
</p>

<h3 id="goroot">Moved GOROOT</h3>

<p><!-- CL 42533 -->
  The <a href="/cmd/go/">go tool</a> will now use the path from which it
  was invoked to attempt to locate the root of the Go install tree.
  This means that if the entire Go installation is moved to a new
  location, the go tool should continue to work as usual.
  This may be overridden by setting <code>GOROOT</code> in the environment,
  which should only be done in unusual circumstances.
  Note that this does not affect the result of
  the <a href="/pkg/runtime/#GOROOT">runtime.GOROOT</a> function, which
  will continue to report the original installation location;
  this may be fixed in later releases.
</p>

<h3 id="compiler">Compiler Toolchain</h3>

<p><!-- CL 37441 -->
  Complex division is now C99-compatible. This has always been the
  case in gccgo and is now fixed in the gc toolchain.
</p>

<p> <!-- CL 36983 -->
  The linker will now generate DWARF information for cgo executables on Windows.
</p>

<p> <!-- CL 44210, CL 40095 -->
  The compiler now includes lexical scopes in the generated DWARF if the
  <code>-N -l</code> flags are provided, allowing
  debuggers to hide variables that are not in scope. The <code>.debug_info</code>
  section is now DWARF version 4.
</p>

<p> <!-- CL 43855 -->
  The values of <code>GOARM</code> and <code>GO386</code> now affect a
  compiled package's build ID, as used by the <code>go</code> tool's
  dependency caching.
</p>

<h3 id="asm">Assembler</h3>

<p> <!-- CL 42028 -->
  The four-operand ARM <code>MULA</code> instruction is now assembled correctly,
  with the addend register as the third argument and the result
  register as the fourth and final argument.
  In previous releases, the two meanings were reversed.
  The three-operand form, in which the fourth argument is implicitly
  the same as the third, is unaffected.
  Code using four-operand <code>MULA</code> instructions
  will need to be updated, but we believe this form is very rarely used.
  <code>MULAWT</code> and <code>MULAWB</code> were already
  using the correct order in all forms and are unchanged.
</p>

<p> <!-- CL 42990 -->
  The assembler now supports <code>ADDSUBPS/PD</code>, completing the
  two missing x86 SSE3 instructions.
</p>

<h3 id="go-doc">Doc</h3>

<p><!-- CL 36031 -->
  Long lists of arguments are now truncated. This improves the readability
  of <code>go</code> <code>doc</code> on some generated code.
</p>

<p><!-- CL 38438 -->
  Viewing documentation on struct fields is now supported.
  For example, <code>go</code> <code>doc</code> <code>http.Client.Jar</code>.
</p>

<h3 id="go-env-json">Env</h3>

<p> <!-- CL 38757 -->
  The new <code>go</code> <code>env</code> <code>-json</code> flag
  enables JSON output, instead of the default OS-specific output
  format.
</p>

<h3 id="go-test-list">Test</h3>

<p> <!-- CL 41195 -->
  The <a href="/cmd/go/#hdr-Description_of_testing_flags"><code>go</code> <code>test</code></a>
  command accepts a new <code>-list</code> flag, which takes a regular
  expression as an argument and prints to stdout the name of any
  tests, benchmarks, or examples that match it, without running them.
</p>


<h3 id="go-tool-pprof">Pprof</h3>

<p> <!-- CL 34192 -->
  Profiles produced by the <code>runtime/pprof</code> package now
  include symbol information, so they can be viewed
  in <code>go</code> <code>tool</code> <code>pprof</code>
  without the binary that produced the profile.
</p>

<p> <!-- CL 38343 -->
  The <code>go</code> <code>tool</code> <code>pprof</code> command now
  uses the HTTP proxy information defined in the environment, using
  <a href="/pkg/net/http/#ProxyFromEnvironment"><code>http.ProxyFromEnvironment</code></a>.
</p>

<h3 id="vet">Vet</h3>

<!-- CL 40112 -->
<p>
  The <a href="/cmd/vet/"><code>vet</code> command</a>
  has been better integrated into the
  <a href="/cmd/go/"><code>go</code> tool</a>,
  so <code>go</code> <code>vet</code> now supports all standard build
  flags while <code>vet</code>'s own flags are now available
  from <code>go</code> <code>vet</code> as well as
  from <code>go</code> <code>tool</code> <code>vet</code>.
</p>

<h3 id="gccgo">Gccgo</h3>

<p>
Due to the alignment of Go's semiannual release schedule with GCC's
annual release schedule,
GCC release 7 contains the Go 1.8.3 version of gccgo.
We expect that the next release, GCC 8, will contain the Go 1.10
version of gccgo.
</p>

<h2 id="runtime">Runtime</h2>

<h3 id="callersframes">Call stacks with inlined frames</h3>

<p>
  Users of
  <a href="/pkg/runtime#Callers"><code>runtime.Callers</code></a>
  should avoid directly inspecting the resulting PC slice and instead use
  <a href="/pkg/runtime#CallersFrames"><code>runtime.CallersFrames</code></a>
  to get a complete view of the call stack, or
  <a href="/pkg/runtime#Caller"><code>runtime.Caller</code></a>
  to get information about a single caller.
  This is because an individual element of the PC slice cannot account
  for inlined frames or other nuances of the call stack.
</p>

<p>
  Specifically, code that directly iterates over the PC slice and uses
  functions such as
  <a href="/pkg/runtime#FuncForPC"><code>runtime.FuncForPC</code></a>
  to resolve each PC individually will miss inlined frames.
  To get a complete view of the stack, such code should instead use
  <code>CallersFrames</code>.
  Likewise, code should not assume that the length returned by
  <code>Callers</code> is any indication of the call depth.
  It should instead count the number of frames returned by
  <code>CallersFrames</code>.
</p>

<p>
  Code that queries a single caller at a specific depth should use
  <code>Caller</code> rather than passing a slice of length 1 to
  <code>Callers</code>.
</p>

<p>
  <a href="/pkg/runtime#CallersFrames"><code>runtime.CallersFrames</code></a>
  has been available since Go 1.7, so code can be updated prior to
  upgrading to Go 1.9.
</p>

<h2 id="performance">Performance</h2>

<p>
  As always, the changes are so general and varied that precise
  statements about performance are difficult to make.  Most programs
  should run a bit faster, due to speedups in the garbage collector,
  better generated code, and optimizations in the core library.
</p>

<h3 id="gc">Garbage Collector</h3>

<p> <!-- CL 37520 -->
  Library functions that used to trigger stop-the-world garbage
  collection now trigger concurrent garbage collection.

  Specifically, <a href="/pkg/runtime/#GC"><code>runtime.GC</code></a>,
  <a href="/pkg/runtime/debug/#SetGCPercent"><code>debug.SetGCPercent</code></a>,
  and
  <a href="/pkg/runtime/debug/#FreeOSMemory"><code>debug.FreeOSMemory</code></a>,
  now trigger concurrent garbage collection, blocking only the calling
  goroutine until the garbage collection is done.
</p>

<p> <!-- CL 34103, CL 39835 -->
  The
  <a href="/pkg/runtime/debug/#SetGCPercent"><code>debug.SetGCPercent</code></a>
  function only triggers a garbage collection if one is immediately
  necessary because of the new GOGC value.
  This makes it possible to adjust GOGC on-the-fly.
</p>

<p> <!-- CL 38732 -->
  Large object allocation performance is significantly improved in
  applications using large (&gt;50GB) heaps containing many large
  objects.
</p>

<p> <!-- CL 34937 -->
  The <a href="/pkg/runtime/#ReadMemStats"><code>runtime.ReadMemStats</code></a>
  function now takes less than 100µs even for very large heaps.
</p>

<h2 id="library">Core library</h2>

<h3 id="monotonic-time">Transparent Monotonic Time support</h3>

<p> <!-- CL 36255 -->
  The <a href="/pkg/time/"><code>time</code></a> package now transparently
  tracks monotonic time in each <a href="/pkg/time/#Time"><code>Time</code></a>
  value, making computing durations between two <code>Time</code> values
  a safe operation in the presence of wall clock adjustments.
  See the <a href="/pkg/time/#hdr-Monotonic_Clocks">package docs</a> and
  <a href="https://golang.org/design/12914-monotonic">design document</a>
  for details.
</p>

<h3 id="math-bits">New bit manipulation package</h3>

<p> <!-- CL 36315 -->
  Go 1.9 includes a new package,
  <a href="/pkg/math/bits/"><code>math/bits</code></a>, with optimized
  implementations for manipulating bits. On most architectures,
  functions in this package are additionally recognized by the
  compiler and treated as intrinsics for additional performance.
</p>

<h3 id="test-helper">Test Helper Functions</h3>

<p> <!-- CL 38796 -->
  The
  new <a href="/pkg/testing/#T.Helper"><code>(*T).Helper</code></a>
  and <a href="/pkg/testing/#B.Helper"><code>(*B).Helper</code></a>
  methods mark the calling function as a test helper function.  When
  printing file and line information, that function will be skipped.
  This permits writing test helper functions while still having useful
  line numbers for users.
</p>

<h3 id="sync-map">Concurrent Map</h3>

<p> <!-- CL 36617 -->
  The new <a href="/pkg/sync/#Map"><code>Map</code></a> type
  in the <a href="/pkg/sync/"><code>sync</code></a> package
  is a concurrent map with amortized-constant-time loads, stores, and
  deletes. It is safe for multiple goroutines to call a <code>Map</code>'s methods
  concurrently.
</p>

<h3 id="pprof-labels">Profiler Labels</h3>

<p><!-- CL 34198 -->
  The <a href="/pkg/runtime/pprof"><code>runtime/pprof</code> package</a>
  now supports adding labels to <code>pprof</code> profiler records.
  Labels form a key-value map that is used to distinguish calls of the
  same function in different contexts when looking at profiles
  with the <a href="/cmd/pprof/"><code>pprof</code> command</a>.
  The <code>pprof</code> package's
  new <a href="/pkg/runtime/pprof/#Do"><code>Do</code> function</a>
  runs code associated with some provided labels. Other new functions
  in the package help work with labels.
</p>

</dl><!-- runtime/pprof -->


<h3 id="minor_library_changes">Minor changes to the library</h3>

<p>
  As always, there are various minor changes and updates to the library,
  made with the Go 1 <a href="/doc/go1compat">promise of compatibility</a>
  in mind.
</p>

<dl id="archive/zip"><dt><a href="/pkg/archive/zip/">archive/zip</a></dt>
  <dd>
    <p><!-- CL 39570 -->
      The
      ZIP <a href="/pkg/archive/zip/#Writer"><code>Writer</code></a>
      now sets the UTF-8 bit in
      the <a href="/pkg/archive/zip/#FileHeader.Flags"><code>FileHeader.Flags</code></a>
      when appropriate.
    </p>

</dl><!-- archive/zip -->

<dl id="crypto/rand"><dt><a href="/pkg/crypto/rand/">crypto/rand</a></dt>
  <dd>
    <p><!-- CL 43852 -->
      On Linux, Go now calls the <code>getrandom</code> system call
      without the <code>GRND_NONBLOCK</code> flag; it will now block
      until the kernel has sufficient randomness. On kernels predating
      the <code>getrandom</code> system call, Go continues to read
      from <code>/dev/urandom</code>.
    </p>

</dl><!-- crypto/rand -->

<dl id="crypto/x509"><dt><a href="/pkg/crypto/x509/">crypto/x509</a></dt>
  <dd>
    <p><!-- CL 36093 -->

      On Unix systems the environment
      variables <code>SSL_CERT_FILE</code>
      and <code>SSL_CERT_DIR</code> can now be used to override the
      system default locations for the SSL certificate file and SSL
      certificate files directory, respectively.
    </p>

    <p>The FreeBSD file <code>/usr/local/etc/ssl/cert.pem</code> is
      now included in the certificate search path.
    </p>

    <p><!-- CL 36900 -->

      The package now supports excluded domains in name constraints.
      In addition to enforcing such constraints,
      <a href="/pkg/crypto/x509/#CreateCertificate"><code>CreateCertificate</code></a>
      will create certificates with excluded name constraints
      if the provided template certificate has the new
      field
      <a href="/pkg/crypto/x509/#Certificate.ExcludedDNSDomains"><code>ExcludedDNSDomains</code></a>
      populated.
    </p>

    <p><!-- CL 36696 -->

    If any SAN extension, including with no DNS names, is present
    in the certificate, then the Common Name from
    <a href="/pkg/crypto/x509/#Certificate.Subject"><code>Subject</code></a> is ignored.
    In previous releases, the code tested only whether DNS-name SANs were
    present in a certificate.
    </p>

</dl><!-- crypto/x509 -->

<dl id="database/sql"><dt><a href="/pkg/database/sql/">database/sql</a></dt>
  <dd>
    <p><!-- CL 35476 -->
      The package will now use a cached <a href="/pkg/database/sql/#Stmt"><code>Stmt</code></a> if
      available in <a href="/pkg/database/sql/#Tx.Stmt"><code>Tx.Stmt</code></a>.
      This prevents statements from being re-prepared each time
      <a href="/pkg/database/sql/#Tx.Stmt"><code>Tx.Stmt</code></a> is called.
    </p>

    <p><!-- CL 38533 -->
      The package now allows drivers to implement their own argument checkers by implementing
      <a href="/pkg/database/sql/driver/#NamedValueChecker"><code>driver.NamedValueChecker</code></a>.
      This also allows drivers to support <code>OUTPUT</code> and <code>INOUT</code> parameter types.
      <a href="/pkg/database/sql/#Out"><code>Out</code></a> should be used to return output parameters
      when supported by the driver.
    </p>

    <p><!-- CL 39031 -->
      <a href="/pkg/database/sql/#Rows.Scan"><code>Rows.Scan</code></a> can now scan user-defined string types.
      Previously the package supported scanning into numeric types like <code>type</code> <code>Int</code> <code>int64</code>. It now also supports
      scanning into string types like <code>type</code> <code>String</code> <code>string</code>.
    </p>

    <p><!-- CL 40694 -->
      The new <a href="/pkg/database/sql/#DB.Conn"><code>DB.Conn</code></a> method returns the new
      <a href="/pkg/database/sql/#Conn"><code>Conn</code></a> type representing an
      exclusive connection to the database from the connection pool. All queries run on
      a <a href="/pkg/database/sql/#Conn"><code>Conn</code></a> will use the same underlying
      connection until <a href="/pkg/database/sql/#Conn.Close"><code>Conn.Close</code></a> is called
      to return the connection to the connection pool.
    </p>

</dl><!-- database/sql -->

<dl id="encoding/asn1"><dt><a href="/pkg/encoding/asn1/">encoding/asn1</a></dt>
  <dd>
    <p><!-- CL 38660 -->
	  The new
	  <a href="/pkg/encoding/asn1/#NullBytes"><code>NullBytes</code></a>
	  and
	  <a href="/pkg/encoding/asn1/#NullRawValue"><code>NullRawValue</code></a>
	  represent the ASN.1 NULL type.
    </p>

</dl><!-- encoding/asn1 -->

<dl id="encoding/base32"><dt><a href="/pkg/encoding/base32/">encoding/base32</a></dt>
  <dd>
    <p><!-- CL 38634 -->
	  The new <a href="/pkg/encoding/base32/#Encoding.WithPadding">Encoding.WithPadding</a>
	  method adds support for custom padding characters and disabling padding.
    </p>

</dl><!-- encoding/base32 -->

<dl id="encoding/csv"><dt><a href="/pkg/encoding/csv/">encoding/csv</a></dt>
  <dd>
    <p><!-- CL 41730 -->
      The new field
      <a href="/pkg/encoding/csv/#Reader.ReuseRecord"><code>Reader.ReuseRecord</code></a>
      controls whether calls to
      <a href="/pkg/encoding/csv/#Reader.Read"><code>Read</code></a>
      may return a slice sharing the backing array of the previous
      call's returned slice for improved performance.
    </p>

</dl><!-- encoding/csv -->

<dl id="fmt"><dt><a href="/pkg/fmt/">fmt</a></dt>
  <dd>
    <p><!-- CL 37051 -->
      The sharp flag ('<code>#</code>') is now supported when printing
      floating point and complex numbers. It will always print a
      decimal point
      for <code>%e</code>, <code>%E</code>, <code>%f</code>, <code>%F</code>, <code>%g</code>
      and <code>%G</code>; it will not remove trailing zeros
      for <code>%g</code> and <code>%G</code>.
    </p>

</dl><!-- fmt -->

<dl id="hash/fnv"><dt><a href="/pkg/hash/fnv/">hash/fnv</a></dt>
  <dd>
    <p><!-- CL 38356 -->
      The package now includes 128-bit FNV-1 and FNV-1a hash support with
      <a href="/pkg/hash/fnv/#New128"><code>New128</code></a> and
      <a href="/pkg/hash/fnv/#New128a"><code>New128a</code></a>, respectively.
    </p>

</dl><!-- hash/fnv -->

<dl id="html/template"><dt><a href="/pkg/html/template/">html/template</a></dt>
  <dd>
    <p><!-- CL 37880, CL 40936 -->
	  The package now reports an error if a predefined escaper (one of
	  "html", "urlquery" and "js") is found in a pipeline and does not match
	  what the auto-escaper would have decided on its own.
	  This avoids certain security or correctness issues.
	  Now use of one of these escapers is always either a no-op or an error.
	  (The no-op case eases migration from <a href="/pkg/text/template/">text/template</a>.)
    </p>

</dl><!-- html/template -->

<dl id="image"><dt><a href="/pkg/image/">image</a></dt>
  <dd>
    <p><!-- CL 36734 -->
	  The <a href="/pkg/image/#Rectangle.Intersect"><code>Rectangle.Intersect</code></a>
	  method now returns a zero <code>Rectangle</code> when called on
	  adjacent but non-overlapping rectangles, as documented. In
	  earlier releases it would incorrectly return an empty but
	  non-zero <code>Rectangle</code>.
    </p>

</dl><!-- image -->

<dl id="image/color"><dt><a href="/pkg/image/color/">image/color</a></dt>
  <dd>
    <p><!-- CL 36732 -->
	  The YCbCr to RGBA conversion formula has been tweaked to ensure
	  that rounding adjustments span the complete [0, 0xffff] RGBA
	  range.
    </p>

</dl><!-- image/color -->

<dl id="image/png"><dt><a href="/pkg/image/png/">image/png</a></dt>
  <dd>
    <p><!-- CL 34150 -->
	  The new <a href="/pkg/image/png/#Encoder.BufferPool"><code>Encoder.BufferPool</code></a>
	  field allows specifying an <a href="/pkg/image/png/#EncoderBufferPool"><code>EncoderBufferPool</code></a>,
	  that will be used by the encoder to get temporary <code>EncoderBuffer</code>
	  buffers when encoding a PNG image.

	  The use of a <code>BufferPool</code> reduces the number of
	  memory allocations performed while encoding multiple images.
    </p>

    <p><!-- CL 38271 -->
	  The package now supports the decoding of transparent 8-bit
	  grayscale ("Gray8") images.
    </p>

</dl><!-- image/png -->

<dl id="math/big"><dt><a href="/pkg/math/big/">math/big</a></dt>
  <dd>
    <p><!-- CL 36487 -->
      The new
      <a href="/pkg/math/big/#Int.IsInt64"><code>IsInt64</code></a>
      and
      <a href="/pkg/math/big/#Int.IsUint64"><code>IsUint64</code></a>
      methods report whether an <code>Int</code>
      may be represented as an <code>int64</code> or <code>uint64</code>
      value.
    </p>

</dl><!-- math/big -->

<dl id="mime/multipart"><dt><a href="/pkg/mime/multipart/">mime/multipart</a></dt>
  <dd>
    <p><!-- CL 39223 -->
      The new
      <a href="/pkg/mime/multipart/#FileHeader.Size"><code>FileHeader.Size</code></a>
      field describes the size of a file in a multipart message.
    </p>

</dl><!-- mime/multipart -->

<dl id="net"><dt><a href="/pkg/net/">net</a></dt>
  <dd>
    <p><!-- CL 32572 -->
      The new
      <a href="/pkg/net/#Resolver.StrictErrors"><code>Resolver.StrictErrors</code></a>
      provides control over how Go's built-in DNS resolver handles
      temporary errors during queries composed of multiple sub-queries,
      such as an A+AAAA address lookup.
    </p>

    <p><!-- CL 37260 -->
      The new
      <a href="/pkg/net/#Resolver.Dial"><code>Resolver.Dial</code></a>
      allows a <code>Resolver</code> to use a custom dial function.
    </p>

    <p><!-- CL 40510 -->
      <a href="/pkg/net/#JoinHostPort"><code>JoinHostPort</code></a> now only places an address in square brackets if the host contains a colon.
      In previous releases it would also wrap addresses in square brackets if they contained a percent ('<code>%</code>') sign.
    </p>

    <p><!-- CL 37913 -->
      The new methods
      <a href="/pkg/net/#TCPConn.SyscallConn"><code>TCPConn.SyscallConn</code></a>,
      <a href="/pkg/net/#IPConn.SyscallConn"><code>IPConn.SyscallConn</code></a>,
      <a href="/pkg/net/#UDPConn.SyscallConn"><code>UDPConn.SyscallConn</code></a>,
      and
      <a href="/pkg/net/#UnixConn.SyscallConn"><code>UnixConn.SyscallConn</code></a>
      provide access to the connections' underlying file descriptors.
    </p>

    <p><!-- 45088 -->
      It is now safe to call <a href="/pkg/net/#Dial"><code>Dial</code></a> with the address obtained from
      <code>(*TCPListener).String()</code> after creating the listener with
      <code><a href="/pkg/net/#Listen">Listen</a>("tcp", ":0")</code>.
      Previously it failed on some machines with half-configured IPv6 stacks.
    </p>

</dl><!-- net -->

<dl id="net/http"><dt><a href="/pkg/net/http/">net/http</a></dt>
  <dd>

    <p><!-- CL 37328 -->
      The <a href="/pkg/net/http/#Cookie.String"><code>Cookie.String</code></a> method, used for
      <code>Cookie</code> and <code>Set-Cookie</code> headers, now encloses values in double quotes
      if the value contains either a space or a comma.
    </p>

    <p>Server changes:</p>
    <ul>
      <li><!-- CL 38194 -->
        <a href="/pkg/net/http/#ServeMux"><code>ServeMux</code></a> now ignores ports in the host
        header when matching handlers. The host is matched unmodified for <code>CONNECT</code> requests.
      </li>

      <li><!-- CL 44074 -->
        The new <a href="/pkg/net/http/#Server.ServeTLS"><code>Server.ServeTLS</code></a> method wraps
        <a href="/pkg/net/http/#Server.Serve"><code>Server.Serve</code></a> with added TLS support.
      </li>

      <li><!-- CL 34727 -->
        <a href="/pkg/net/http/#Server.WriteTimeout"><code>Server.WriteTimeout</code></a>
        now applies to HTTP/2 connections and is enforced per-stream.
      </li>

      <li><!-- CL 43231 -->
        HTTP/2 now uses the priority write scheduler by default.
        Frames are scheduled by following HTTP/2 priorities as described in
        <a href="https://tools.ietf.org/html/rfc7540#section-5.3">RFC 7540 Section 5.3</a>.
      </li>

      <li><!-- CL 36483 -->
        The HTTP handler returned by <a href="/pkg/net/http/#StripPrefix"><code>StripPrefix</code></a>
        now calls its provided handler with a modified clone of the original <code>*http.Request</code>.
        Any code storing per-request state in maps keyed by <code>*http.Request</code> should
        use
        <a href="/pkg/net/http/#Request.Context"><code>Request.Context</code></a>,
        <a href="/pkg/net/http/#Request.WithContext"><code>Request.WithContext</code></a>,
        and
        <a href="/pkg/context/#WithValue"><code>context.WithValue</code></a> instead.
      </li>

      <li><!-- CL 35490 -->
        <a href="/pkg/net/http/#LocalAddrContextKey"><code>LocalAddrContextKey</code></a> now contains
        the connection's actual network address instead of the interface address used by the listener.
      </li>
    </ul>

    <p>Client &amp; Transport changes:</p>
    <ul>
      <li><!-- CL 35488 -->
        The <a href="/pkg/net/http/#Transport"><code>Transport</code></a>
        now supports making requests via SOCKS5 proxy when the URL returned by
        <a href="/pkg/net/http/#Transport.Proxy"><code>Transport.Proxy</code></a>
        has the scheme <code>socks5</code>.
      </li>
    </ul>

</dl><!-- net/http -->

<dl id="net/http/fcgi"><dt><a href="/pkg/net/http/fcgi/">net/http/fcgi</a></dt>
  <dd>
    <p><!-- CL 40012 -->
      The new
      <a href="/pkg/net/http/fcgi/#ProcessEnv"><code>ProcessEnv</code></a>
      function returns FastCGI environment variables associated with an HTTP request
      for which there are no appropriate
      <a href="/pkg/net/http/#Request"><code>http.Request</code></a>
      fields, such as <code>REMOTE_USER</code>.
    </p>

</dl><!-- net/http/fcgi -->

<dl id="net/http/httptest"><dt><a href="/pkg/net/http/httptest/">net/http/httptest</a></dt>
  <dd>
    <p><!-- CL 34639 -->
      The new
      <a href="/pkg/net/http/httptest/#Server.Client"><code>Server.Client</code></a>
      method returns an HTTP client configured for making requests to the test server.
    </p>

    <p>
      The new
      <a href="/pkg/net/http/httptest/#Server.Certificate"><code>Server.Certificate</code></a>
      method returns the test server's TLS certificate, if any.
    </p>

</dl><!-- net/http/httptest -->

<dl id="net/http/httputil"><dt><a href="/pkg/net/http/httputil/">net/http/httputil</a></dt>
  <dd>
    <p><!-- CL 43712 -->
      The <a href="/pkg/net/http/httputil/#ReverseProxy"><code>ReverseProxy</code></a>
      now proxies all HTTP/2 response trailers, even those not declared in the initial response
      header. Such undeclared trailers are used by the gRPC protocol.
    </p>

</dl><!-- net/http/httputil -->

<dl id="os"><dt><a href="/pkg/os/">os</a></dt>
  <dd>
    <p><!-- CL 36800 -->
      The <code>os</code> package now uses the internal runtime poller
      for file I/O.
      This reduces the number of threads required for read/write
      operations on pipes, and it eliminates races when one goroutine
      closes a file while another is using the file for I/O.
    </p>

  <dd>
    <p><!-- CL 37915 -->
      On Windows,
      <a href="/pkg/os/#Args"><code>Args</code></a>
      is now populated without <code>shell32.dll</code>, improving process start-up time by 1-7 ms.
      </p>

</dl><!-- os -->

<dl id="os/exec"><dt><a href="/pkg/os/exec/">os/exec</a></dt>
  <dd>
    <p><!-- CL 37586 -->
      The <code>os/exec</code> package now prevents child processes from being created with
      any duplicate environment variables.
      If <a href="/pkg/os/exec/#Cmd.Env"><code>Cmd.Env</code></a>
      contains duplicate environment keys, only the last
      value in the slice for each duplicate key is used.
    </p>

</dl><!-- os/exec -->

<dl id="os/user"><dt><a href="/pkg/os/user/">os/user</a></dt>
  <dd>
    <p><!-- CL 37664 -->
      <a href="/pkg/os/user/#Lookup"><code>Lookup</code></a> and
      <a href="/pkg/os/user/#LookupId"><code>LookupId</code></a> now
      work on Unix systems when <code>CGO_ENABLED=0</code> by reading
      the <code>/etc/passwd</code> file.
    </p>

    <p><!-- CL 33713 -->
      <a href="/pkg/os/user/#LookupGroup"><code>LookupGroup</code></a> and
      <a href="/pkg/os/user/#LookupGroupId"><code>LookupGroupId</code></a> now
      work on Unix systems when <code>CGO_ENABLED=0</code> by reading
      the <code>/etc/group</code> file.
    </p>

</dl><!-- os/user -->

<dl id="reflect"><dt><a href="/pkg/reflect/">reflect</a></dt>
  <dd>
    <p><!-- CL 38335 -->
      The new
      <a href="/pkg/reflect/#MakeMapWithSize"><code>MakeMapWithSize</code></a>
      function creates a map with a capacity hint.
    </p>

</dl><!-- reflect -->

<dl id="runtime"><dt><a href="/pkg/runtime/">runtime</a></dt>
  <dd>
    <p><!-- CL 37233, CL 37726 -->
      Tracebacks generated by the runtime and recorded in profiles are
      now accurate in the presence of inlining.
      To retrieve tracebacks programmatically, applications should use
      <a href="/pkg/runtime/#CallersFrames"><code>runtime.CallersFrames</code></a>
      rather than directly iterating over the results of
      <a href="/pkg/runtime/#Callers"><code>runtime.Callers</code></a>.
    </p>

    <p><!-- CL 38403 -->
      On Windows, Go no longer forces the system timer to run at high
      resolution when the program is idle.
      This should reduce the impact of Go programs on battery life.
    </p>

    <p><!-- CL 29341 -->
      On FreeBSD, <code>GOMAXPROCS</code> and
      <a href="/pkg/runtime/#NumCPU"><code>runtime.NumCPU</code></a>
      are now based on the process' CPU mask, rather than the total
      number of CPUs.
    </p>

    <p><!-- CL 43641 -->
      The runtime has preliminary support for Android O.
    </p>

</dl><!-- runtime -->

<dl id="runtime/debug"><dt><a href="/pkg/runtime/debug/">runtime/debug</a></dt>
  <dd>
    <p><!-- CL 34013 -->
      Calling
      <a href="/pkg/runtime/debug/#SetGCPercent"><code>SetGCPercent</code></a>
      with a negative value no longer runs an immediate garbage collection.
    </p>

</dl><!-- runtime/debug -->

<dl id="runtime/trace"><dt><a href="/pkg/runtime/trace/">runtime/trace</a></dt>
  <dd>
    <p><!-- CL 36015 -->
      The execution trace now displays mark assist events, which
      indicate when an application goroutine is forced to assist
      garbage collection because it is allocating too quickly.
    </p>

    <p><!-- CL 40810 -->
      "Sweep" events now encompass the entire process of finding free
      space for an allocation, rather than recording each individual
      span that is swept.
      This reduces allocation latency when tracing allocation-heavy
      programs.
      The sweep event shows how many bytes were swept and how many
      were reclaimed.
    </p>

</dl><!-- runtime/trace -->

<dl id="sync"><dt><a href="/pkg/sync/">sync</a></dt>
  <dd>
    <p><!-- CL 34310 -->
      <a href="/pkg/sync/#Mutex"><code>Mutex</code></a> is now more fair.
    </p>

</dl><!-- sync -->

<dl id="syscall"><dt><a href="/pkg/syscall/">syscall</a></dt>
  <dd>
    <p><!-- CL 36697 -->
      The new field
      <a href="/pkg/syscall/#Credential.NoSetGroups"><code>Credential.NoSetGroups</code></a>
      controls whether Unix systems make a <code>setgroups</code> system call
      to set supplementary groups when starting a new process.
    </p>

    <p><!-- CL 43512 -->
      The new field
      <a href="/pkg/syscall/#SysProcAttr.AmbientCaps"><code>SysProcAttr.AmbientCaps</code></a>
      allows setting ambient capabilities on Linux 4.3+ when creating
      a new process.
    </p>

    <p><!-- CL 37439 -->
      On 64-bit x86 Linux, process creation latency has been optimized with
      use of <code>CLONE_VFORK</code> and <code>CLONE_VM</code>.
    </p>

    <p><!-- CL 37913 -->
      The new
      <a href="/pkg/syscall/#Conn"><code>Conn</code></a>
      interface describes some types in the
      <a href="/pkg/net/"><code>net</code></a>
      package that can provide access to their underlying file descriptor
      using the new
      <a href="/pkg/syscall/#RawConn"><code>RawConn</code></a>
      interface.
    </p>

</dl><!-- syscall -->


<dl id="testing/quick"><dt><a href="/pkg/testing/quick/">testing/quick</a></dt>
  <dd>
    <p><!-- CL 39152 -->
      The package now chooses values in the full range when
      generating <code>int64</code> and <code>uint64</code> random
      numbers; in earlier releases generated values were always
      limited to the [-2<sup>62</sup>, 2<sup>62</sup>) range.
    </p>

    <p>
      In previous releases, using a nil
      <a href="/pkg/testing/quick/#Config.Rand"><code>Config.Rand</code></a>
      value caused a fixed deterministic random number generator to be used.
      It now uses a random number generator seeded with the current time.
      For the old behavior, set <code>Config.Rand</code> to <code>rand.New(rand.NewSource(0))</code>.
    </p>

</dl><!-- testing/quick -->

<dl id="text/template"><dt><a href="/pkg/text/template/">text/template</a></dt>
  <dd>
    <p><!-- CL 38420 -->
	  The handling of empty blocks, which was broken by a Go 1.8
	  change that made the result dependent on the order of templates,
	  has been fixed, restoring the old Go 1.7 behavior.
    </p>

</dl><!-- text/template -->

<dl id="time"><dt><a href="/pkg/time/">time</a></dt>
  <dd>
    <p><!-- CL 36615 -->
      The new methods
      <a href="/pkg/time/#Duration.Round"><code>Duration.Round</code></a>
      and
      <a href="/pkg/time/#Duration.Truncate"><code>Duration.Truncate</code></a>
      handle rounding and truncating durations to multiples of a given duration.
    </p>

    <p><!-- CL 35710 -->
      Retrieving the time and sleeping now work correctly under Wine.
    </p>

    <p>
      If a <code>Time</code> value has a monotonic clock reading, its
      string representation (as returned by <code>String</code>) now includes a
      final field <code>"m=±value"</code>, where <code>value</code> is the
      monotonic clock reading formatted as a decimal number of seconds.
    </p>

    <p><!-- CL 44832 -->
      The included <code>tzdata</code> timezone database has been
      updated to version 2017b. As always, it is only used if the
      system does not already have the database available.
    </p>

</dl><!-- time -->
