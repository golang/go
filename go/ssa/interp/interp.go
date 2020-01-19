// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ssa/interp defines an interpreter for the SSA
// representation of Go programs.
//
// This interpreter is provided as an adjunct for testing the SSA
// construction algorithm.  Its purpose is to provide a minimal
// metacircular implementation of the dynamic semantics of each SSA
// instruction.  It is not, and will never be, a production-quality Go
// interpreter.
//
// The following is a partial list of Go features that are currently
// unsupported or incomplete in the interpreter.
//
// * Unsafe operations, including all uses of unsafe.Pointer, are
// impossible to support given the "boxed" value representation we
// have chosen.
//
// * The reflect package is only partially implemented.
//
// * The "testing" package is no longer supported because it
// depends on low-level details that change too often.
//
// * "sync/atomic" operations are not atomic due to the "boxed" value
// representation: it is not possible to read, modify and write an
// interface value atomically. As a consequence, Mutexes are currently
// broken.
//
// * recover is only partially implemented.  Also, the interpreter
// makes no attempt to distinguish target panics from interpreter
// crashes.
//
// * the sizes of the int, uint and uintptr types in the target
// program are assumed to be the same as those of the interpreter
// itself.
//
// * all values occupy space, even those of types defined by the spec
// to have zero size, e.g. struct{}.  This can cause asymptotic
// performance degradation.
//
// * os.Exit is implemented using panic, causing deferred functions to
// run.
package interp // import "golang.org/x/tools/go/ssa/interp"

import (
	"fmt"
	"go/token"
	"go/types"
	"os"
	"reflect"
	"runtime"
	"sync/atomic"

	"golang.org/x/tools/go/ssa"
)

type continuation int

const (
	kNext continuation = iota
	kReturn
	kJump
)

// Mode is a bitmask of options affecting the interpreter.
type Mode uint

const (
	DisableRecover Mode = 1 << iota // Disable recover() in target programs; show interpreter crash instead.
	EnableTracing                   // Print a trace of all instructions as they are interpreted.
)

type methodSet map[string]*ssa.Function

// State shared between all interpreted goroutines.
type interpreter struct {
	osArgs             []value              // the value of os.Args
	prog               *ssa.Program         // the SSA program
	globals            map[ssa.Value]*value // addresses of global variables (immutable)
	mode               Mode                 // interpreter options
	reflectPackage     *ssa.Package         // the fake reflect package
	errorMethods       methodSet            // the method set of reflect.error, which implements the error interface.
	rtypeMethods       methodSet            // the method set of rtype, which implements the reflect.Type interface.
	runtimeErrorString types.Type           // the runtime.errorString type
	sizes              types.Sizes          // the effective type-sizing function
	goroutines         int32                // atomically updated
}

type deferred struct {
	fn    value
	args  []value
	instr *ssa.Defer
	tail  *deferred
}

type frame struct {
	i                *interpreter
	caller           *frame
	fn               *ssa.Function
	block, prevBlock *ssa.BasicBlock
	env              map[ssa.Value]value // dynamic values of SSA variables
	locals           []value
	defers           *deferred
	result           value
	panicking        bool
	panic            interface{}
}

func (fr *frame) get(key ssa.Value) value {
	switch key := key.(type) {
	case nil:
		// Hack; simplifies handling of optional attributes
		// such as ssa.Slice.{Low,High}.
		return nil
	case *ssa.Function, *ssa.Builtin:
		return key
	case *ssa.Const:
		return constValue(key)
	case *ssa.Global:
		if r, ok := fr.i.globals[key]; ok {
			return r
		}
	}
	if r, ok := fr.env[key]; ok {
		return r
	}
	panic(fmt.Sprintf("get: no value for %T: %v", key, key.Name()))
}

// runDefer runs a deferred call d.
// It always returns normally, but may set or clear fr.panic.
//
func (fr *frame) runDefer(d *deferred) {
	if fr.i.mode&EnableTracing != 0 {
		fmt.Fprintf(os.Stderr, "%s: invoking deferred function call\n",
			fr.i.prog.Fset.Position(d.instr.Pos()))
	}
	var ok bool
	defer func() {
		if !ok {
			// Deferred call created a new state of panic.
			fr.panicking = true
			fr.panic = recover()
		}
	}()
	call(fr.i, fr, d.instr.Pos(), d.fn, d.args)
	ok = true
}

// runDefers executes fr's deferred function calls in LIFO order.
//
// On entry, fr.panicking indicates a state of panic; if
// true, fr.panic contains the panic value.
//
// On completion, if a deferred call started a panic, or if no
// deferred call recovered from a previous state of panic, then
// runDefers itself panics after the last deferred call has run.
//
// If there was no initial state of panic, or it was recovered from,
// runDefers returns normally.
//
func (fr *frame) runDefers() {
	for d := fr.defers; d != nil; d = d.tail {
		fr.runDefer(d)
	}
	fr.defers = nil
	if fr.panicking {
		panic(fr.panic) // new panic, or still panicking
	}
}

// lookupMethod returns the method set for type typ, which may be one
// of the interpreter's fake types.
func lookupMethod(i *interpreter, typ types.Type, meth *types.Func) *ssa.Function {
	switch typ {
	case rtypeType:
		return i.rtypeMethods[meth.Id()]
	case errorType:
		return i.errorMethods[meth.Id()]
	}
	return i.prog.LookupMethod(typ, meth.Pkg(), meth.Name())
}

// visitInstr interprets a single ssa.Instruction within the activation
// record frame.  It returns a continuation value indicating where to
// read the next instruction from.
func visitInstr(fr *frame, instr ssa.Instruction) continuation {
	switch instr := instr.(type) {
	case *ssa.DebugRef:
		// no-op

	case *ssa.UnOp:
		fr.env[instr] = unop(instr, fr.get(instr.X))

	case *ssa.BinOp:
		fr.env[instr] = binop(instr.Op, instr.X.Type(), fr.get(instr.X), fr.get(instr.Y))

	case *ssa.Call:
		fn, args := prepareCall(fr, &instr.Call)
		fr.env[instr] = call(fr.i, fr, instr.Pos(), fn, args)

	case *ssa.ChangeInterface:
		fr.env[instr] = fr.get(instr.X)

	case *ssa.ChangeType:
		fr.env[instr] = fr.get(instr.X) // (can't fail)

	case *ssa.Convert:
		fr.env[instr] = conv(instr.Type(), instr.X.Type(), fr.get(instr.X))

	case *ssa.MakeInterface:
		fr.env[instr] = iface{t: instr.X.Type(), v: fr.get(instr.X)}

	case *ssa.Extract:
		fr.env[instr] = fr.get(instr.Tuple).(tuple)[instr.Index]

	case *ssa.Slice:
		fr.env[instr] = slice(fr.get(instr.X), fr.get(instr.Low), fr.get(instr.High), fr.get(instr.Max))

	case *ssa.Return:
		switch len(instr.Results) {
		case 0:
		case 1:
			fr.result = fr.get(instr.Results[0])
		default:
			var res []value
			for _, r := range instr.Results {
				res = append(res, fr.get(r))
			}
			fr.result = tuple(res)
		}
		fr.block = nil
		return kReturn

	case *ssa.RunDefers:
		fr.runDefers()

	case *ssa.Panic:
		panic(targetPanic{fr.get(instr.X)})

	case *ssa.Send:
		fr.get(instr.Chan).(chan value) <- fr.get(instr.X)

	case *ssa.Store:
		store(deref(instr.Addr.Type()), fr.get(instr.Addr).(*value), fr.get(instr.Val))

	case *ssa.If:
		succ := 1
		if fr.get(instr.Cond).(bool) {
			succ = 0
		}
		fr.prevBlock, fr.block = fr.block, fr.block.Succs[succ]
		return kJump

	case *ssa.Jump:
		fr.prevBlock, fr.block = fr.block, fr.block.Succs[0]
		return kJump

	case *ssa.Defer:
		fn, args := prepareCall(fr, &instr.Call)
		fr.defers = &deferred{
			fn:    fn,
			args:  args,
			instr: instr,
			tail:  fr.defers,
		}

	case *ssa.Go:
		fn, args := prepareCall(fr, &instr.Call)
		atomic.AddInt32(&fr.i.goroutines, 1)
		go func() {
			call(fr.i, nil, instr.Pos(), fn, args)
			atomic.AddInt32(&fr.i.goroutines, -1)
		}()

	case *ssa.MakeChan:
		fr.env[instr] = make(chan value, asInt(fr.get(instr.Size)))

	case *ssa.Alloc:
		var addr *value
		if instr.Heap {
			// new
			addr = new(value)
			fr.env[instr] = addr
		} else {
			// local
			addr = fr.env[instr].(*value)
		}
		*addr = zero(deref(instr.Type()))

	case *ssa.MakeSlice:
		slice := make([]value, asInt(fr.get(instr.Cap)))
		tElt := instr.Type().Underlying().(*types.Slice).Elem()
		for i := range slice {
			slice[i] = zero(tElt)
		}
		fr.env[instr] = slice[:asInt(fr.get(instr.Len))]

	case *ssa.MakeMap:
		reserve := 0
		if instr.Reserve != nil {
			reserve = asInt(fr.get(instr.Reserve))
		}
		fr.env[instr] = makeMap(instr.Type().Underlying().(*types.Map).Key(), reserve)

	case *ssa.Range:
		fr.env[instr] = rangeIter(fr.get(instr.X), instr.X.Type())

	case *ssa.Next:
		fr.env[instr] = fr.get(instr.Iter).(iter).next()

	case *ssa.FieldAddr:
		fr.env[instr] = &(*fr.get(instr.X).(*value)).(structure)[instr.Field]

	case *ssa.Field:
		fr.env[instr] = fr.get(instr.X).(structure)[instr.Field]

	case *ssa.IndexAddr:
		x := fr.get(instr.X)
		idx := fr.get(instr.Index)
		switch x := x.(type) {
		case []value:
			fr.env[instr] = &x[asInt(idx)]
		case *value: // *array
			fr.env[instr] = &(*x).(array)[asInt(idx)]
		default:
			panic(fmt.Sprintf("unexpected x type in IndexAddr: %T", x))
		}

	case *ssa.Index:
		fr.env[instr] = fr.get(instr.X).(array)[asInt(fr.get(instr.Index))]

	case *ssa.Lookup:
		fr.env[instr] = lookup(instr, fr.get(instr.X), fr.get(instr.Index))

	case *ssa.MapUpdate:
		m := fr.get(instr.Map)
		key := fr.get(instr.Key)
		v := fr.get(instr.Value)
		switch m := m.(type) {
		case map[value]value:
			m[key] = v
		case *hashmap:
			m.insert(key.(hashable), v)
		default:
			panic(fmt.Sprintf("illegal map type: %T", m))
		}

	case *ssa.TypeAssert:
		fr.env[instr] = typeAssert(fr.i, instr, fr.get(instr.X).(iface))

	case *ssa.MakeClosure:
		var bindings []value
		for _, binding := range instr.Bindings {
			bindings = append(bindings, fr.get(binding))
		}
		fr.env[instr] = &closure{instr.Fn.(*ssa.Function), bindings}

	case *ssa.Phi:
		for i, pred := range instr.Block().Preds {
			if fr.prevBlock == pred {
				fr.env[instr] = fr.get(instr.Edges[i])
				break
			}
		}

	case *ssa.Select:
		var cases []reflect.SelectCase
		if !instr.Blocking {
			cases = append(cases, reflect.SelectCase{
				Dir: reflect.SelectDefault,
			})
		}
		for _, state := range instr.States {
			var dir reflect.SelectDir
			if state.Dir == types.RecvOnly {
				dir = reflect.SelectRecv
			} else {
				dir = reflect.SelectSend
			}
			var send reflect.Value
			if state.Send != nil {
				send = reflect.ValueOf(fr.get(state.Send))
			}
			cases = append(cases, reflect.SelectCase{
				Dir:  dir,
				Chan: reflect.ValueOf(fr.get(state.Chan)),
				Send: send,
			})
		}
		chosen, recv, recvOk := reflect.Select(cases)
		if !instr.Blocking {
			chosen-- // default case should have index -1.
		}
		r := tuple{chosen, recvOk}
		for i, st := range instr.States {
			if st.Dir == types.RecvOnly {
				var v value
				if i == chosen && recvOk {
					// No need to copy since send makes an unaliased copy.
					v = recv.Interface().(value)
				} else {
					v = zero(st.Chan.Type().Underlying().(*types.Chan).Elem())
				}
				r = append(r, v)
			}
		}
		fr.env[instr] = r

	default:
		panic(fmt.Sprintf("unexpected instruction: %T", instr))
	}

	// if val, ok := instr.(ssa.Value); ok {
	// 	fmt.Println(toString(fr.env[val])) // debugging
	// }

	return kNext
}

// prepareCall determines the function value and argument values for a
// function call in a Call, Go or Defer instruction, performing
// interface method lookup if needed.
//
func prepareCall(fr *frame, call *ssa.CallCommon) (fn value, args []value) {
	v := fr.get(call.Value)
	if call.Method == nil {
		// Function call.
		fn = v
	} else {
		// Interface method invocation.
		recv := v.(iface)
		if recv.t == nil {
			panic("method invoked on nil interface")
		}
		if f := lookupMethod(fr.i, recv.t, call.Method); f == nil {
			// Unreachable in well-typed programs.
			panic(fmt.Sprintf("method set for dynamic type %v does not contain %s", recv.t, call.Method))
		} else {
			fn = f
		}
		args = append(args, recv.v)
	}
	for _, arg := range call.Args {
		args = append(args, fr.get(arg))
	}
	return
}

// call interprets a call to a function (function, builtin or closure)
// fn with arguments args, returning its result.
// callpos is the position of the callsite.
//
func call(i *interpreter, caller *frame, callpos token.Pos, fn value, args []value) value {
	switch fn := fn.(type) {
	case *ssa.Function:
		if fn == nil {
			panic("call of nil function") // nil of func type
		}
		return callSSA(i, caller, callpos, fn, args, nil)
	case *closure:
		return callSSA(i, caller, callpos, fn.Fn, args, fn.Env)
	case *ssa.Builtin:
		return callBuiltin(caller, callpos, fn, args)
	}
	panic(fmt.Sprintf("cannot call %T", fn))
}

func loc(fset *token.FileSet, pos token.Pos) string {
	if pos == token.NoPos {
		return ""
	}
	return " at " + fset.Position(pos).String()
}

// callSSA interprets a call to function fn with arguments args,
// and lexical environment env, returning its result.
// callpos is the position of the callsite.
//
func callSSA(i *interpreter, caller *frame, callpos token.Pos, fn *ssa.Function, args []value, env []value) value {
	if i.mode&EnableTracing != 0 {
		fset := fn.Prog.Fset
		// TODO(adonovan): fix: loc() lies for external functions.
		fmt.Fprintf(os.Stderr, "Entering %s%s.\n", fn, loc(fset, fn.Pos()))
		suffix := ""
		if caller != nil {
			suffix = ", resuming " + caller.fn.String() + loc(fset, callpos)
		}
		defer fmt.Fprintf(os.Stderr, "Leaving %s%s.\n", fn, suffix)
	}
	fr := &frame{
		i:      i,
		caller: caller, // for panic/recover
		fn:     fn,
	}
	if fn.Parent() == nil {
		name := fn.String()
		if ext := externals[name]; ext != nil {
			if i.mode&EnableTracing != 0 {
				fmt.Fprintln(os.Stderr, "\t(external)")
			}
			return ext(fr, args)
		}
		if fn.Blocks == nil {
			panic("no code for function: " + name)
		}
	}
	fr.env = make(map[ssa.Value]value)
	fr.block = fn.Blocks[0]
	fr.locals = make([]value, len(fn.Locals))
	for i, l := range fn.Locals {
		fr.locals[i] = zero(deref(l.Type()))
		fr.env[l] = &fr.locals[i]
	}
	for i, p := range fn.Params {
		fr.env[p] = args[i]
	}
	for i, fv := range fn.FreeVars {
		fr.env[fv] = env[i]
	}
	for fr.block != nil {
		runFrame(fr)
	}
	// Destroy the locals to avoid accidental use after return.
	for i := range fn.Locals {
		fr.locals[i] = bad{}
	}
	return fr.result
}

// runFrame executes SSA instructions starting at fr.block and
// continuing until a return, a panic, or a recovered panic.
//
// After a panic, runFrame panics.
//
// After a normal return, fr.result contains the result of the call
// and fr.block is nil.
//
// A recovered panic in a function without named return parameters
// (NRPs) becomes a normal return of the zero value of the function's
// result type.
//
// After a recovered panic in a function with NRPs, fr.result is
// undefined and fr.block contains the block at which to resume
// control.
//
func runFrame(fr *frame) {
	defer func() {
		if fr.block == nil {
			return // normal return
		}
		if fr.i.mode&DisableRecover != 0 {
			return // let interpreter crash
		}
		fr.panicking = true
		fr.panic = recover()
		if fr.i.mode&EnableTracing != 0 {
			fmt.Fprintf(os.Stderr, "Panicking: %T %v.\n", fr.panic, fr.panic)
		}
		fr.runDefers()
		fr.block = fr.fn.Recover
	}()

	for {
		if fr.i.mode&EnableTracing != 0 {
			fmt.Fprintf(os.Stderr, ".%s:\n", fr.block)
		}
	block:
		for _, instr := range fr.block.Instrs {
			if fr.i.mode&EnableTracing != 0 {
				if v, ok := instr.(ssa.Value); ok {
					fmt.Fprintln(os.Stderr, "\t", v.Name(), "=", instr)
				} else {
					fmt.Fprintln(os.Stderr, "\t", instr)
				}
			}
			switch visitInstr(fr, instr) {
			case kReturn:
				return
			case kNext:
				// no-op
			case kJump:
				break block
			}
		}
	}
}

// doRecover implements the recover() built-in.
func doRecover(caller *frame) value {
	// recover() must be exactly one level beneath the deferred
	// function (two levels beneath the panicking function) to
	// have any effect.  Thus we ignore both "defer recover()" and
	// "defer f() -> g() -> recover()".
	if caller.i.mode&DisableRecover == 0 &&
		caller != nil && !caller.panicking &&
		caller.caller != nil && caller.caller.panicking {
		caller.caller.panicking = false
		p := caller.caller.panic
		caller.caller.panic = nil

		// TODO(adonovan): support runtime.Goexit.
		switch p := p.(type) {
		case targetPanic:
			// The target program explicitly called panic().
			return p.v
		case runtime.Error:
			// The interpreter encountered a runtime error.
			return iface{caller.i.runtimeErrorString, p.Error()}
		case string:
			// The interpreter explicitly called panic().
			return iface{caller.i.runtimeErrorString, p}
		default:
			panic(fmt.Sprintf("unexpected panic type %T in target call to recover()", p))
		}
	}
	return iface{}
}

// setGlobal sets the value of a system-initialized global variable.
func setGlobal(i *interpreter, pkg *ssa.Package, name string, v value) {
	if g, ok := i.globals[pkg.Var(name)]; ok {
		*g = v
		return
	}
	panic("no global variable: " + pkg.Pkg.Path() + "." + name)
}

// Interpret interprets the Go program whose main package is mainpkg.
// mode specifies various interpreter options.  filename and args are
// the initial values of os.Args for the target program.  sizes is the
// effective type-sizing function for this program.
//
// Interpret returns the exit code of the program: 2 for panic (like
// gc does), or the argument to os.Exit for normal termination.
//
// The SSA program must include the "runtime" package.
//
func Interpret(mainpkg *ssa.Package, mode Mode, sizes types.Sizes, filename string, args []string) (exitCode int) {
	i := &interpreter{
		prog:       mainpkg.Prog,
		globals:    make(map[ssa.Value]*value),
		mode:       mode,
		sizes:      sizes,
		goroutines: 1,
	}
	runtimePkg := i.prog.ImportedPackage("runtime")
	if runtimePkg == nil {
		panic("ssa.Program doesn't include runtime package")
	}
	i.runtimeErrorString = runtimePkg.Type("errorString").Object().Type()

	initReflect(i)

	i.osArgs = append(i.osArgs, filename)
	for _, arg := range args {
		i.osArgs = append(i.osArgs, arg)
	}

	for _, pkg := range i.prog.AllPackages() {
		// Initialize global storage.
		for _, m := range pkg.Members {
			switch v := m.(type) {
			case *ssa.Global:
				cell := zero(deref(v.Type()))
				i.globals[v] = &cell
			}
		}
	}

	// Top-level error handler.
	exitCode = 2
	defer func() {
		if exitCode != 2 || i.mode&DisableRecover != 0 {
			return
		}
		switch p := recover().(type) {
		case exitPanic:
			exitCode = int(p)
			return
		case targetPanic:
			fmt.Fprintln(os.Stderr, "panic:", toString(p.v))
		case runtime.Error:
			fmt.Fprintln(os.Stderr, "panic:", p.Error())
		case string:
			fmt.Fprintln(os.Stderr, "panic:", p)
		default:
			fmt.Fprintf(os.Stderr, "panic: unexpected type: %T: %v\n", p, p)
		}

		// TODO(adonovan): dump panicking interpreter goroutine?
		// buf := make([]byte, 0x10000)
		// runtime.Stack(buf, false)
		// fmt.Fprintln(os.Stderr, string(buf))
		// (Or dump panicking target goroutine?)
	}()

	// Run!
	call(i, nil, token.NoPos, mainpkg.Func("init"), nil)
	if mainFn := mainpkg.Func("main"); mainFn != nil {
		call(i, nil, token.NoPos, mainFn, nil)
		exitCode = 0
	} else {
		fmt.Fprintln(os.Stderr, "No main function.")
		exitCode = 1
	}
	return
}

// deref returns a pointer's element type; otherwise it returns typ.
// TODO(adonovan): Import from ssa?
func deref(typ types.Type) types.Type {
	if p, ok := typ.Underlying().(*types.Pointer); ok {
		return p.Elem()
	}
	return typ
}
