# Trace experiments

Execution traces allow for trialing new events on an experimental basis via
trace experiments.
This document is a guide that explains how you can define your own trace
experiments.

Note that if you're just trying to do some debugging or perform some light
instrumentation, then a trace experiment is way overkill.
Use `runtime/trace.Log` instead.
Even if you're just trying to create a proof-of-concept for a low-frequency
event, `runtime/trace.Log` will probably be easier overall if you can make
it work.

Consider a trace experiment if:
- The volume of new trace events will be relatively high, and so the events
  would benefit from a more compact representation (creating new tables to
  deduplicate data, taking advantage of the varint representation, etc.).
- It's not safe to call `runtime/trace.Log` (or its runtime equivalent) in
  the contexts you want to generate an event (for example, for events about
  timers).

## Defining a new experiment

To define a new experiment, modify `internal/trace/tracev2` to define a
new `Experiment` enum value.

An experiment consists of two parts: timed events and experimental batches.
Timed events are events like any other and follow the same format.
They are easier to order and require less work to make use of.
Experimental batches are essentially bags of bytes that correspond to
an entire trace generation.
What they contain and how they're interpreted is totally up to you, but
they're most often useful for tables that your other events can refer into.
For example, the AllocFree experiment uses them to store type information
that allocation events can refer to.

### Defining new events

1. Define your new experiment event types (by convention, experimental events
   types start at ID 127, so look for the `const` block defining events
   starting there).
2. Describe your new events in `specs`.
   Use the documentation for `Spec` to write your new specs, and check your
   work by running the tests in the `internal/trace/tracev2` package.
   If you wish for your event argument to be interpreted in a particular
   way, follow the naming convention in
   `src/internal/trace/tracev2/spec.go`.
   For example, if you intend to emit a string argument, make sure the
   argument name has the suffix `string`.
3. Add ordering and validation logic for your new events to
   `src/internal/trace/order.go` by listing handlers for those events in
   the `orderingDispatch` table.
   If your events are always emitted in a regular user goroutine context,
   then the handler should be trivial and just validate the scheduling
   context to match userGoReqs.
   If it's more complicated, see `(*ordering).advanceAllocFree` for a
   slightly more complicated example that handles events from a larger
   variety of execution environments.
   If you need to encode a partial ordering, look toward the scheduler
   events (names beginning with `Go`) or just ask someone for help.
4. Add your new events to the `tracev2Type2Kind` table in
   `src/internal/trace/event.go`.

## Emitting data

### Emitting your new events

1. Define helper methods on `runtime.traceEventWriter` for emitting your
   events.
2. Instrument the runtime with calls to these helper methods.
   Make sure to call `traceAcquire` and `traceRelease` around the operation
   your event represents, otherwise it will not be emitted atomically with
   that operation completing, resulting in a potentially misleading trace.

### Emitting experimental batches

To emit experimental batches, use the `runtime.unsafeTraceExpWriter` to
write experimental batches associated with your experiment.
Heed the warnings and make sure that while you write them, the trace
generation cannot advance.
Note that each experiment can only have one distinguishable set of
batches.

## Recovering experimental data

### Recovering experimental events from the trace

Experimental events will appear in the event stream as an event with the
`EventExperimental` `Kind`.
Use the `Experimental` method to collect the raw data inserted into the
trace.
It's essentially up to you to interpret the event from here.
I recommend writing a thin wrapper API to present a cleaner interface if you
so desire.

### Recovering experimental batches

Parse out all the experimental batches from `Sync` events as they come.
These experimental batches are all for the same generation as all the
experimental events up until the next `Sync` event.
