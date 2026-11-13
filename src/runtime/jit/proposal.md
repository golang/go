<!-- Copyright 2026 The Go Authors. All rights reserved. -->
# Proposal: runtime/jit — cooperative generated frames and ABI helpers

Status: experimental, requires GOEXPERIMENT=jit. Discussion: https://go.dev/issue/78189.
The public package is shipped with the runtime; descriptions and generated code
are specific to that toolchain and architecture. This is not a stable machine ABI.

## Frame registration and precise GC

Register(Region) registers one nonoverlapping executable range. StackMaps identify
exact return-PC safepoints and contain copied pointer masks plus declarative
caller PC/SP/BP recovery recipes. AddStackMaps publishes append-only batches
before their code becomes callable. GC scanning and stack copying execute no
client callbacks. Every enclosing Go frame must remain reachable, including
through consecutive generated frames. Stack copying relocates mapped pointers
and saved frame pointers. UnwindStop is a traceback barrier, not permission to
hide live enclosing Go frames from GC.

ValidateRegion checks address ranges, PC ordering, bitmap size/alignment and
unwind-offset arithmetic. Registration and append reject malformed public input
with MetadataError before publication. Runtime stack-bound violations report the
PC, region, SP and offending map range before terminating. Validation cannot
infer omitted live pointers or prove that machine code matches its maps.
Handle.Snapshot and Handle.WriteTo provide independent diagnostic snapshots and
readable maps without invoking Describe.

Addressable JIT stack objects are described by their individual live pointer
fields. Those fields are roots at the safepoint even if the containing object is
otherwise unreachable. Reachability-dependent JIT stack-object descriptors are
not part of this version. Mapped fields must be initialized and must be cleared
or removed from subsequent maps before their slots are reused. Nonescaping
closures may borrow their owning invocation's stack and may be called by Go
synchronously; they must not escape to another invocation or goroutine.
References to ordinary Go stack objects use Go's normal reachability processing.

## ABI descriptions

GetCallingConventionForFunc accepts a function value, including a typed nil;
GetCallingConventionForType accepts a reflect.Type for a function not created yet.
Each original input/output retains its Go type and a list of ValueParts. Parts
identify their original byte offset, width, GC-pointer role, and physical register
or caller-SP-relative stack location. Empty values retain parameter identity.
Aggregate assignment rolls back completely if either register bank is exhausted.
The description includes stack results, register-argument spill homes, total call
area, link-area and entry-SP conventions, register roles and call clobbers.

RegisterConfig supplies architectural integer/FP argument banks and allocation
constraints. It does not select live values, interference colors, eviction costs,
or a client's temporary-register budget. Clients retain their allocators.
Argument registers are fixed only at the corresponding entry/call/return boundary.
Descriptions are prepared once per function/signature and may be cached as
immutable generator metadata. Querying a signature is a compilation/setup task,
not a required operation at every emitted call. MemCP's generator prepares static
Go entries at package initialization and reads cached argument/result locations,
pointer masks and call-frame sizes while emitting. Its common func(...Scmer)
Scmer closure convention is prepared once and used at entry, call and return
boundaries. The client retains its own register allocator and buffer writer.

Interface values are supported in these descriptions. An empty interface currently
contains a type-metadata word and a data-pointer word; a nonempty interface uses
an itab word instead of the type word. The data word is not an arbitrary unboxed
scalar. JITs which directly inspect those words depend on that representation.
ABI helpers abstract placement, not a future fundamentally different value layout.
No general-purpose interface-construction emitter is included.

GetMemoryLayout describes scalar memory components, padding-free offsets and
GC-pointer roles independently of call locations. It covers strings, slices,
interfaces and aggregate fields. Store, Copy and Clear are typed Go-call helpers
which retain the running compiler's write barriers. EmitWriteBarrierCheck allows
a safepoint-free plain-store path while barriers are disabled; its enabled path
must perform the required typed writes. In particular, a pointer-free new value
does not eliminate the deletion barrier for an overwritten old pointer.

## Code emission

CodeBuffer exposes Offset, Write and Patch over caller-owned unpublished memory.
Write is atomic per supplied block. Insufficient capacity panics with
CodeBufferFull {Offset, Required, Available}; no partial block is written.
Recover does not resume the emitter. The client restarts its compilation unit,
discarding bytes, roots, labels and metadata from the failed attempt. Patch only
changes previously written bytes. Emitters return opaque fixups and exact return
PCs, not raw patch pointers. Automatic segment chaining is not included.

EmitFrame/PatchFrame, EmitReturn/EmitLeave/EmitLeafReturn, EmitStackCheck and
EmitStackGrow own the runtime-sensitive frame sequences. EmitCall handles direct,
register-indirect and Go-closure calls. EmitCallFrame/ReleaseCallFrame provide the
signature-specific outgoing area. Clients prepare arguments, preserve live
clobbered registers, and combine their local liveness with FrameStackMap and the
slow-path maps returned by the helpers before publishing code. Emitters build a
complete sequence before writing it; they do not execute per-byte callbacks in
generated code. Caller-owned mappings retain their existing permission,
instruction-cache, publication and lifetime responsibilities.

FunctionHeader is the runtime-owned closure-context prefix. Function[F] turns
such a typed context into an ordinary Go-callable function; FunctionContext
retrieves the context of either a generated or normal Go function value.
EmitLoadFunctionContext loads that context from a Go function variable, so a
client need not encode the func-value representation itself.
Application objects such as MemCP Proc/lambda structures remain client-owned.
TailTypeFor/Alloc retain the precisely typed variable-tail allocation facility;
its possible separation remains a proposal-review question, not a new general
untyped allocator API.

ABI query/emission helpers currently support amd64, arm64 and riscv64. Region
registration retains the additional architectures covered by its existing tests.

## Safepoints, preemption and callbacks

Arbitrary JIT instructions, partial prologues and pointer write-barrier sequences
are not asynchronous safepoints. The runtime rejects async preemption and debugger
call injection at unknown Go functions. Signal tracebacks at unmapped PCs may be
incomplete; this does not make such PCs legal GC stops. Conservative provenance
from injected Go register-save frames propagates to the immediate described JIT
parent and to Go stack objects reached conservatively.

Preempt reports a pending request; Yield is an actual scheduling boundary.
EmitPreemptCheck/PatchPreemptCheck optionally emit a branch to a client-owned
slow path, and EmitYield emits the scheduling call with its exact return-PC and
clobbers. Clients preserve live values and publish the corresponding stack map.
The package never inserts checks or selects polling frequency automatically.
Every unbounded native loop must promptly observe requests and reach Yield or an
equivalent actual scheduling boundary through a described call. Calls to arbitrary
nosplit leaves and returning alone do not guarantee a yield. Blocking foreign code
still requires scheduler transitions such as cgo. This API is for cooperative,
Go-compatible generated code; it does not replace the cgo infrastructure for
opaque foreign execution.

Bounded kernels may complete their work before returning to their surrounding
Go/operator boundary. MemCP deliberately inserts no cooperative polls, budgets,
cancellation exits or Yield calls into its short expression kernels or fused
shard kernels. A shard kernel completes its shard before returning. Ordinary Go
helper calls retain their normal Go GC/scheduler semantics. Other JIT consumers
choose their own policy; this consumer policy is not imposed by runtime/jit.

Normal Go functions called by JIT code may allocate and block normally. Describe
is a separate runtime-context symbolization callback and must not allocate or grow
its stack. Pointer scanning and relocation never invoke it. Globals and embedded
pointer constants must remain rooted in typed Go-managed owners. Generated heap
pointer writes require write barriers or typed Go helpers, including writes to
freshly allocated closures. Recoverable hardware faults at generated PCs remain
outside the specified contract.

## Compatibility and validation

Recompute descriptions and regenerate machine code with each toolchain build;
there is no cross-version serialized-code format. The experiment may change its
public API in response to runtime review. Disabled builds do not expose the
package; disabled frame lookup/scanning/relocation paths are compile-time gated.
The existing registration, region, stack-map and allocation signatures remain
source-compatible with the previous fork. The ABI descriptions and emitters are
optional additions: existing JITs may continue to supply their own emitters and
valid declarative maps. Metadata validation intentionally rejects malformed
recipes; unused bitmap padding bits remain ignored for compatibility.

Tests independently hand-emit integer and FP constant returns and closure-context
loads, then invoke them from compiled Go. Other tests exercise generated calls to
Go, ordinary Go closure calls, mixed integer/FP arguments, forced stack growth,
GC pointer retention, nested frames, panic/recover, map publication, diagnostic
copies and rejected metadata. A native loop without regular Go callbacks tests
cooperative scheduling and GC progress. Cross-compilation is not execution on the
target architecture and is reported separately.
Additional execution tests cover stack-only aggregate arguments/results, custom
frame registers across stack growth, function-variable loads, both write-barrier
branch states, and an independent complete Linux/amd64 consumer example.
