This directory contains test inputs for the bisect command.

Each text file is a txtar archive (see <https://pkg.go.dev/golang.org/x/tools/txtar>
or `go doc txtar`).

The comment at the top of the archive is a JSON object describing a
target behavior. Specifically, the Fail key gives a boolean expression
that should provoke a failure. Bisect's job is to discover this
condition.

The Bisect key describes settings in the Bisect struct that we want to
change, to simulate the use of various command-line options.

The txtar archive files should be "stdout" and "stderr", giving the
expected standard output and standard error. If the bisect command
should exit with a non-zero status, the stderr in the archive will end
with the line "<bisect failed>".

Running `go test -update` will rewrite the stdout and stderr files in
each testdata archive to match the current state of the tool. This is
a useful command when the logging prints from bisect change or when
writing a new test.

To use `go test -update` to write a new test:

 - Create a new .txt file with just a JSON object at the top,
   specifying what you want to test.
 - Run `go test -update`.
 - Reload the .txt file and read the stdout and stderr to see if you agree.
