This is a description of the profile.proto format.

# Overview

Profile.proto is a data representation for profile data. It is independent of
the type of data being collected and the sampling process used to collect that
data. On disk, it is represented as a gzip-compressed protocol buffer, described
at src/proto/profile.proto

A profile in this context refers to a collection of samples, each one
representing measurements performed at a certain point in the life of a job. A
sample associates a set of measurement values with a list of locations, commonly
representing the program call stack when the sample was taken.

Tools such as pprof analyze these samples and display this information in
multiple forms, such as identifying hottest locations, building graphical call
graphs or trees, etc.

# General structure of a profile

A profile is represented on a Profile message, which contain the following
fields:

* *sample*: A profile sample, with the values measured and the associated call
  stack as a list of location ids. Samples with identical call stacks can be
  merged by adding their respective values, element by element.
* *location*: A unique place in the program, commonly mapped to a single
  instruction address. It has a unique nonzero id, to be referenced from the
  samples. It contains source information in the form of lines, and a mapping id
  that points to a binary.
* *function*: A program function as defined in the program source. It has a
  unique nonzero id, referenced from the location lines. It contains a
  human-readable name for the function (eg a C++ demangled name), a system name
  (eg a C++ mangled name), the name of the corresponding source file, and other
  function attributes.
* *mapping*: A binary that is part of the program during the profile
  collection. It has a unique nonzero id, referenced from the locations. It
  includes details on how the binary was mapped during program execution. By
  convention the main program binary is the first mapping, followed by any
  shared libraries.
* *string_table*: All strings in the profile are represented as indices into
  this repeating field. The first string is empty, so index == 0 always
  represents the empty string.

# Measurement values

Measurement values are represented as 64-bit integers. The profile contains an
explicit description of each value represented, using a ValueType message, with
two fields:

* *Type*: A human-readable description of the type semantics. For example “cpu”
  to represent CPU time, “wall” or “time” for wallclock time, or “memory” for
  bytes allocated.
* *Unit*: A human-readable name of the unit represented by the 64-bit integer
  values. For example, it could be “nanoseconds” or “milliseconds” for a time
  value, or “bytes” or “megabytes” for a memory size. If this is just
  representing a number of events, the recommended unit name is “count”.

A profile can represent multiple measurements per sample, but all samples must
have the same number and type of measurements. The actual values are stored in
the Sample.value fields, each one described by the corresponding
Profile.sample_type field.

Some profiles have a uniform period that describe the granularity of the data
collection. For example, a CPU profile may have a period of 100ms, or a memory
allocation profile may have a period of 512kb. Profiles can optionally describe
such a value on the Profile.period and Profile.period_type fields. The profile
period is meant for human consumption and does not affect the interpretation of
the profiling data.

By convention, the first value on all profiles is the number of samples
collected at this call stack, with unit “count”. Because the profile does not
describe the sampling process beyond the optional period, it must include
unsampled values for all measurements. For example, a CPU profile could have
value[0] == samples, and value[1] == time in milliseconds.

## Locations, functions and mappings

Each sample lists the id of each location where the sample was collected, in
bottom-up order. Each location has an explicit unique nonzero integer id,
independent of its position in the profile, and holds additional information to
identify the corresponding source.

The profile source is expected to perform any adjustment required to the
locations in order to point to the calls in the stack. For example, if the
profile source extracts the call stack by walking back over the program stack,
it must adjust the instruction addresses to point to the actual call
instruction, instead of the instruction that each call will return to.

Sources usually generate profiles that fall into these two categories:

* *Unsymbolized profiles*: These only contain instruction addresses, and are to
  be symbolized by a separate tool. It is critical for each location to point to
  a valid mapping, which will provide the information required for
  symbolization. These are used for profiles of compiled languages, such as C++
  and Go.

* *Symbolized profiles*: These contain all the symbol information available for
  the profile. Mappings and instruction addresses are optional for symbolized
  locations. These are used for profiles of interpreted or jitted languages,
  such as Java or Python.  Also, the profile format allows the generation of
  mixed profiles, with symbolized and unsymbolized locations.

The symbol information is represented in the repeating lines field of the
Location message. A location has multiple lines if it reflects multiple program
sources, for example if representing inlined call stacks. Lines reference
functions by their unique nonzero id, and the source line number within the
source file listed by the function. A function contains the source attributes
for a function, including its name, source file, etc. Functions include both a
user and a system form of the name, for example to include C++ demangled and
mangled names. For profiles where only a single name exists, both should be set
to the same string.

Mappings are also referenced from locations by their unique nonzero id, and
include all information needed to symbolize addresses within the mapping. It
includes similar information to the Linux /proc/self/maps file. Locations
associated to a mapping should have addresses that land between the mapping
start and limit. Also, if available, mappings should include a build id to
uniquely identify the version of the binary being used.

## Labels

Samples optionally contain labels, which are annotations to discriminate samples
with identical locations. For example, a label can be used on a malloc profile
to indicate allocation size, so two samples on the same call stack with sizes
2MB and 4MB do not get merged into a single sample with two allocations and a
size of 6MB.

Labels can be string-based or numeric. They are represented by the Label
message, with a key identifying the label and either a string or numeric
value. For numeric labels, by convention the key represents the measurement unit
of the numeric value. So for the previous example, the samples would have labels
{“bytes”, 2097152} and {“bytes”, 4194304}.

## Keep and drop expressions

Some profile sources may have knowledge of locations that are uninteresting or
irrelevant. However, if symbolization is needed in order to identify these
locations, the profile source may not be able to remove them when the profile is
generated. The profile format provides a mechanism to identify these frames by
name, through regular expressions.

These expressions must match the function name in its entirety. Frames that
match Profile.drop\_frames will be dropped from the profile, along with any
frames below it. Frames that match Profile.keep\_frames will be kept, even if
they match drop\_frames.

