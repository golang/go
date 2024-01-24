// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package slog provides structured logging,
in which log records include a message,
a severity level, and various other attributes
expressed as key-value pairs.

It defines a type, [Logger],
which provides several methods (such as [Logger.Info] and [Logger.Error])
for reporting events of interest.

Each Logger is associated with a [Handler].
A Logger output method creates a [Record] from the method arguments
and passes it to the Handler, which decides how to handle it.
There is a default Logger accessible through top-level functions
(such as [Info] and [Error]) that call the corresponding Logger methods.

A log record consists of a time, a level, a message, and a set of key-value
pairs, where the keys are strings and the values may be of any type.
As an example,

	slog.Info("hello", "count", 3)

creates a record containing the time of the call,
a level of Info, the message "hello", and a single
pair with key "count" and value 3.

The [Info] top-level function calls the [Logger.Info] method on the default Logger.
In addition to [Logger.Info], there are methods for Debug, Warn and Error levels.
Besides these convenience methods for common levels,
there is also a [Logger.Log] method which takes the level as an argument.
Each of these methods has a corresponding top-level function that uses the
default logger.

The default handler formats the log record's message, time, level, and attributes
as a string and passes it to the [log] package.

	2022/11/08 15:28:26 INFO hello count=3

For more control over the output format, create a logger with a different handler.
This statement uses [New] to create a new logger with a [TextHandler]
that writes structured records in text form to standard error:

	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))

[TextHandler] output is a sequence of key=value pairs, easily and unambiguously
parsed by machine. This statement:

	logger.Info("hello", "count", 3)

produces this output:

	time=2022-11-08T15:28:26.000-05:00 level=INFO msg=hello count=3

The package also provides [JSONHandler], whose output is line-delimited JSON:

	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	logger.Info("hello", "count", 3)

produces this output:

	{"time":"2022-11-08T15:28:26.000000000-05:00","level":"INFO","msg":"hello","count":3}

Both [TextHandler] and [JSONHandler] can be configured with [HandlerOptions].
There are options for setting the minimum level (see Levels, below),
displaying the source file and line of the log call, and
modifying attributes before they are logged.

Setting a logger as the default with

	slog.SetDefault(logger)

will cause the top-level functions like [Info] to use it.
[SetDefault] also updates the default logger used by the [log] package,
so that existing applications that use [log.Printf] and related functions
will send log records to the logger's handler without needing to be rewritten.

Some attributes are common to many log calls.
For example, you may wish to include the URL or trace identifier of a server request
with all log events arising from the request.
Rather than repeat the attribute with every log call, you can use [Logger.With]
to construct a new Logger containing the attributes:

	logger2 := logger.With("url", r.URL)

The arguments to With are the same key-value pairs used in [Logger.Info].
The result is a new Logger with the same handler as the original, but additional
attributes that will appear in the output of every call.

# Levels

A [Level] is an integer representing the importance or severity of a log event.
The higher the level, the more severe the event.
This package defines constants for the most common levels,
but any int can be used as a level.

In an application, you may wish to log messages only at a certain level or greater.
One common configuration is to log messages at Info or higher levels,
suppressing debug logging until it is needed.
The built-in handlers can be configured with the minimum level to output by
setting [HandlerOptions.Level].
The program's `main` function typically does this.
The default value is LevelInfo.

Setting the [HandlerOptions.Level] field to a [Level] value
fixes the handler's minimum level throughout its lifetime.
Setting it to a [LevelVar] allows the level to be varied dynamically.
A LevelVar holds a Level and is safe to read or write from multiple
goroutines.
To vary the level dynamically for an entire program, first initialize
a global LevelVar:

	var programLevel = new(slog.LevelVar) // Info by default

Then use the LevelVar to construct a handler, and make it the default:

	h := slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: programLevel})
	slog.SetDefault(slog.New(h))

Now the program can change its logging level with a single statement:

	programLevel.Set(slog.LevelDebug)

# Groups

Attributes can be collected into groups.
A group has a name that is used to qualify the names of its attributes.
How this qualification is displayed depends on the handler.
[TextHandler] separates the group and attribute names with a dot.
[JSONHandler] treats each group as a separate JSON object, with the group name as the key.

Use [Group] to create a Group attribute from a name and a list of key-value pairs:

	slog.Group("request",
	    "method", r.Method,
	    "url", r.URL)

TextHandler would display this group as

	request.method=GET request.url=http://example.com

JSONHandler would display it as

	"request":{"method":"GET","url":"http://example.com"}

Use [Logger.WithGroup] to qualify all of a Logger's output
with a group name. Calling WithGroup on a Logger results in a
new Logger with the same Handler as the original, but with all
its attributes qualified by the group name.

This can help prevent duplicate attribute keys in large systems,
where subsystems might use the same keys.
Pass each subsystem a different Logger with its own group name so that
potential duplicates are qualified:

	logger := slog.Default().With("id", systemID)
	parserLogger := logger.WithGroup("parser")
	parseInput(input, parserLogger)

When parseInput logs with parserLogger, its keys will be qualified with "parser",
so even if it uses the common key "id", the log line will have distinct keys.

# Contexts

Some handlers may wish to include information from the [context.Context] that is
available at the call site. One example of such information
is the identifier for the current span when tracing is enabled.

The [Logger.Log] and [Logger.LogAttrs] methods take a context as a first
argument, as do their corresponding top-level functions.

Although the convenience methods on Logger (Info and so on) and the
corresponding top-level functions do not take a context, the alternatives ending
in "Context" do. For example,

	slog.InfoContext(ctx, "message")

It is recommended to pass a context to an output method if one is available.

# Attrs and Values

An [Attr] is a key-value pair. The Logger output methods accept Attrs as well as
alternating keys and values. The statement

	slog.Info("hello", slog.Int("count", 3))

behaves the same as

	slog.Info("hello", "count", 3)

There are convenience constructors for [Attr] such as [Int], [String], and [Bool]
for common types, as well as the function [Any] for constructing Attrs of any
type.

The value part of an Attr is a type called [Value].
Like an [any], a Value can hold any Go value,
but it can represent typical values, including all numbers and strings,
without an allocation.

For the most efficient log output, use [Logger.LogAttrs].
It is similar to [Logger.Log] but accepts only Attrs, not alternating
keys and values; this allows it, too, to avoid allocation.

The call

	logger.LogAttrs(ctx, slog.LevelInfo, "hello", slog.Int("count", 3))

is the most efficient way to achieve the same output as

	slog.InfoContext(ctx, "hello", "count", 3)

# Customizing a type's logging behavior

If a type implements the [LogValuer] interface, the [Value] returned from its LogValue
method is used for logging. You can use this to control how values of the type
appear in logs. For example, you can redact secret information like passwords,
or gather a struct's fields in a Group. See the examples under [LogValuer] for
details.

A LogValue method may return a Value that itself implements [LogValuer]. The [Value.Resolve]
method handles these cases carefully, avoiding infinite loops and unbounded recursion.
Handler authors and others may wish to use [Value.Resolve] instead of calling LogValue directly.

# Wrapping output methods

The logger functions use reflection over the call stack to find the file name
and line number of the logging call within the application. This can produce
incorrect source information for functions that wrap slog. For instance, if you
define this function in file mylog.go:

	func Infof(logger *slog.Logger, format string, args ...any) {
	    logger.Info(fmt.Sprintf(format, args...))
	}

and you call it like this in main.go:

	Infof(slog.Default(), "hello, %s", "world")

then slog will report the source file as mylog.go, not main.go.

A correct implementation of Infof will obtain the source location
(pc) and pass it to NewRecord.
The Infof function in the package-level example called "wrapping"
demonstrates how to do this.

# Working with Records

Sometimes a Handler will need to modify a Record
before passing it on to another Handler or backend.
A Record contains a mixture of simple public fields (e.g. Time, Level, Message)
and hidden fields that refer to state (such as attributes) indirectly. This
means that modifying a simple copy of a Record (e.g. by calling
[Record.Add] or [Record.AddAttrs] to add attributes)
may have unexpected effects on the original.
Before modifying a Record, use [Record.Clone] to
create a copy that shares no state with the original,
or create a new Record with [NewRecord]
and build up its Attrs by traversing the old ones with [Record.Attrs].

# Performance considerations

If profiling your application demonstrates that logging is taking significant time,
the following suggestions may help.

If many log lines have a common attribute, use [Logger.With] to create a Logger with
that attribute. The built-in handlers will format that attribute only once, at the
call to [Logger.With]. The [Handler] interface is designed to allow that optimization,
and a well-written Handler should take advantage of it.

The arguments to a log call are always evaluated, even if the log event is discarded.
If possible, defer computation so that it happens only if the value is actually logged.
For example, consider the call

	slog.Info("starting request", "url", r.URL.String())  // may compute String unnecessarily

The URL.String method will be called even if the logger discards Info-level events.
Instead, pass the URL directly:

	slog.Info("starting request", "url", &r.URL) // calls URL.String only if needed

The built-in [TextHandler] will call its String method, but only
if the log event is enabled.
Avoiding the call to String also preserves the structure of the underlying value.
For example [JSONHandler] emits the components of the parsed URL as a JSON object.
If you want to avoid eagerly paying the cost of the String call
without causing the handler to potentially inspect the structure of the value,
wrap the value in a fmt.Stringer implementation that hides its Marshal methods.

You can also use the [LogValuer] interface to avoid unnecessary work in disabled log
calls. Say you need to log some expensive value:

	slog.Debug("frobbing", "value", computeExpensiveValue(arg))

Even if this line is disabled, computeExpensiveValue will be called.
To avoid that, define a type implementing LogValuer:

	type expensive struct { arg int }

	func (e expensive) LogValue() slog.Value {
	    return slog.AnyValue(computeExpensiveValue(e.arg))
	}

Then use a value of that type in log calls:

	slog.Debug("frobbing", "value", expensive{arg})

Now computeExpensiveValue will only be called when the line is enabled.

The built-in handlers acquire a lock before calling [io.Writer.Write]
to ensure that each record is written in one piece. User-defined
handlers are responsible for their own locking.

# Writing a handler

For a guide to writing a custom handler, see https://golang.org/s/slog-handler-guide.
*/
package slog
