<!--{
	"Title": "Data Race Detector",
	"Template": true
}-->

<h2 id="Introduction">Introduction</h2>

<p>
Data races are among the most common and hardest to debug types of bugs in concurrent systems.
A data race occurs when two goroutines access the same variable concurrently and at least one of the accesses is a write.
See the <a href="/ref/mem/">The Go Memory Model</a> for details.
</p>

<p>
Here is an example of a data race that can lead to crashes and memory corruption:
</p>

<pre>
func main() {
	c := make(chan bool)
	m := make(map[string]string)
	go func() {
		m["1"] = "a" // First conflicting access.
		c &lt;- true
	}()
	m["2"] = "b" // Second conflicting access.
	&lt;-c
	for k, v := range m {
		fmt.Println(k, v)
	}
}
</pre>

<h2 id="Usage">Usage</h2>

<p>
To help diagnose such bugs, Go includes a built-in data race detector.
To use it, add the <code>-race</code> flag to the go command:
</p>

<pre>
$ go test -race mypkg    // to test the package
$ go run -race mysrc.go  // to run the source file
$ go build -race mycmd   // to build the command
$ go install -race mypkg // to install the package
</pre>

<h2 id="Report_Format">Report Format</h2>

<p>
When the race detector finds a data race in the program, it prints a report.
The report contains stack traces for conflicting accesses, as well as stacks where the involved goroutines were created.
Here is an example:
</p>

<pre>
WARNING: DATA RACE
Read by goroutine 185:
  net.(*pollServer).AddFD()
      src/net/fd_unix.go:89 +0x398
  net.(*pollServer).WaitWrite()
      src/net/fd_unix.go:247 +0x45
  net.(*netFD).Write()
      src/net/fd_unix.go:540 +0x4d4
  net.(*conn).Write()
      src/net/net.go:129 +0x101
  net.func·060()
      src/net/timeout_test.go:603 +0xaf

Previous write by goroutine 184:
  net.setWriteDeadline()
      src/net/sockopt_posix.go:135 +0xdf
  net.setDeadline()
      src/net/sockopt_posix.go:144 +0x9c
  net.(*conn).SetDeadline()
      src/net/net.go:161 +0xe3
  net.func·061()
      src/net/timeout_test.go:616 +0x3ed

Goroutine 185 (running) created at:
  net.func·061()
      src/net/timeout_test.go:609 +0x288

Goroutine 184 (running) created at:
  net.TestProlongTimeout()
      src/net/timeout_test.go:618 +0x298
  testing.tRunner()
      src/testing/testing.go:301 +0xe8
</pre>

<h2 id="Options">Options</h2>

<p>
The <code>GORACE</code> environment variable sets race detector options.
The format is:
</p>

<pre>
GORACE="option1=val1 option2=val2"
</pre>

<p>
The options are:
</p>

<ul>
<li>
<code>log_path</code> (default <code>stderr</code>): The race detector writes
its report to a file named <code>log_path.<em>pid</em></code>.
The special names <code>stdout</code>
and <code>stderr</code> cause reports to be written to standard output and
standard error, respectively.
</li>

<li>
<code>exitcode</code> (default <code>66</code>): The exit status to use when
exiting after a detected race.
</li>

<li>
<code>strip_path_prefix</code> (default <code>""</code>): Strip this prefix
from all reported file paths, to make reports more concise.
</li>

<li>
<code>history_size</code> (default <code>1</code>): The per-goroutine memory
access history is <code>32K * 2**history_size elements</code>.
Increasing this value can avoid a "failed to restore the stack" error in reports, at the
cost of increased memory usage.
</li>

<li>
<code>halt_on_error</code> (default <code>0</code>): Controls whether the program
exits after reporting first data race.
</li>
</ul>

<p>
Example:
</p>

<pre>
$ GORACE="log_path=/tmp/race/report strip_path_prefix=/my/go/sources/" go test -race
</pre>

<h2 id="Excluding_Tests">Excluding Tests</h2>

<p>
When you build with <code>-race</code> flag, the <code>go</code> command defines additional
<a href="/pkg/go/build/#hdr-Build_Constraints">build tag</a> <code>race</code>.
You can use the tag to exclude some code and tests when running the race detector.
Some examples:
</p>

<pre>
// +build !race

package foo

// The test contains a data race. See issue 123.
func TestFoo(t *testing.T) {
	// ...
}

// The test fails under the race detector due to timeouts.
func TestBar(t *testing.T) {
	// ...
}

// The test takes too long under the race detector.
func TestBaz(t *testing.T) {
	// ...
}
</pre>

<h2 id="How_To_Use">How To Use</h2>

<p>
To start, run your tests using the race detector (<code>go test -race</code>).
The race detector only finds races that happen at runtime, so it can't find
races in code paths that are not executed.
If your tests have incomplete coverage,
you may find more races by running a binary built with <code>-race</code> under a realistic
workload.
</p>

<h2 id="Typical_Data_Races">Typical Data Races</h2>

<p>
Here are some typical data races.  All of them can be detected with the race detector.
</p>

<h3 id="Race_on_loop_counter">Race on loop counter</h3>

<pre>
func main() {
	var wg sync.WaitGroup
	wg.Add(5)
	for i := 0; i < 5; i++ {
		go func() {
			fmt.Println(i) // Not the 'i' you are looking for.
			wg.Done()
		}()
	}
	wg.Wait()
}
</pre>

<p>
The variable <code>i</code> in the function literal is the same variable used by the loop, so
the read in the goroutine races with the loop increment.
(This program typically prints 55555, not 01234.)
The program can be fixed by making a copy of the variable:
</p>

<pre>
func main() {
	var wg sync.WaitGroup
	wg.Add(5)
	for i := 0; i < 5; i++ {
		go func(j int) {
			fmt.Println(j) // Good. Read local copy of the loop counter.
			wg.Done()
		}(i)
	}
	wg.Wait()
}
</pre>

<h3 id="Accidentally_shared_variable">Accidentally shared variable</h3>

<pre>
// ParallelWrite writes data to file1 and file2, returns the errors.
func ParallelWrite(data []byte) chan error {
	res := make(chan error, 2)
	f1, err := os.Create("file1")
	if err != nil {
		res &lt;- err
	} else {
		go func() {
			// This err is shared with the main goroutine,
			// so the write races with the write below.
			_, err = f1.Write(data)
			res &lt;- err
			f1.Close()
		}()
	}
	f2, err := os.Create("file2") // The second conflicting write to err.
	if err != nil {
		res &lt;- err
	} else {
		go func() {
			_, err = f2.Write(data)
			res &lt;- err
			f2.Close()
		}()
	}
	return res
}
</pre>

<p>
The fix is to introduce new variables in the goroutines (note the use of <code>:=</code>):
</p>

<pre>
			...
			_, err := f1.Write(data)
			...
			_, err := f2.Write(data)
			...
</pre>

<h3 id="Unprotected_global_variable">Unprotected global variable</h3>

<p>
If the following code is called from several goroutines, it leads to races on the <code>service</code> map.
Concurrent reads and writes of the same map are not safe:
</p>

<pre>
var service map[string]net.Addr

func RegisterService(name string, addr net.Addr) {
	service[name] = addr
}

func LookupService(name string) net.Addr {
	return service[name]
}
</pre>

<p>
To make the code safe, protect the accesses with a mutex:
</p>

<pre>
var (
	service   map[string]net.Addr
	serviceMu sync.Mutex
)

func RegisterService(name string, addr net.Addr) {
	serviceMu.Lock()
	defer serviceMu.Unlock()
	service[name] = addr
}

func LookupService(name string) net.Addr {
	serviceMu.Lock()
	defer serviceMu.Unlock()
	return service[name]
}
</pre>

<h3 id="Primitive_unprotected_variable">Primitive unprotected variable</h3>

<p>
Data races can happen on variables of primitive types as well (<code>bool</code>, <code>int</code>, <code>int64</code>, etc.),
as in this example:
</p>

<pre>
type Watchdog struct{ last int64 }

func (w *Watchdog) KeepAlive() {
	w.last = time.Now().UnixNano() // First conflicting access.
}

func (w *Watchdog) Start() {
	go func() {
		for {
			time.Sleep(time.Second)
			// Second conflicting access.
			if w.last < time.Now().Add(-10*time.Second).UnixNano() {
				fmt.Println("No keepalives for 10 seconds. Dying.")
				os.Exit(1)
			}
		}
	}()
}
</pre>

<p>
Even such "innocent" data races can lead to hard-to-debug problems caused by
non-atomicity of the memory accesses,
interference with compiler optimizations,
or reordering issues accessing processor memory .
</p>

<p>
A typical fix for this race is to use a channel or a mutex.
To preserve the lock-free behavior, one can also use the
<a href="/pkg/sync/atomic/"><code>sync/atomic</code></a> package.
</p>

<pre>
type Watchdog struct{ last int64 }

func (w *Watchdog) KeepAlive() {
	atomic.StoreInt64(&amp;w.last, time.Now().UnixNano())
}

func (w *Watchdog) Start() {
	go func() {
		for {
			time.Sleep(time.Second)
			if atomic.LoadInt64(&amp;w.last) < time.Now().Add(-10*time.Second).UnixNano() {
				fmt.Println("No keepalives for 10 seconds. Dying.")
				os.Exit(1)
			}
		}
	}()
}
</pre>

<h2 id="Supported_Systems">Supported Systems</h2>

<p>
The race detector runs on <code>darwin/amd64</code>, <code>freebsd/amd64</code>,
<code>linux/amd64</code>, and <code>windows/amd64</code>.
</p>

<h2 id="Runtime_Overheads">Runtime Overhead</h2>

<p>
The cost of race detection varies by program, but for a typical program, memory
usage may increase by 5-10x and execution time by 2-20x.
</p>
