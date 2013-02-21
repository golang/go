// Package exp/ssa/interp defines an interpreter for the SSA
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
// * "sync/atomic" operations are not currently atomic due to the
// "boxed" value representation: it is not possible to read, modify
// and write an interface value atomically.  As a consequence, Mutexes
// are currently broken.  TODO(adonovan): provide a metacircular
// implementation of Mutex avoiding the broken atomic primitives.
//
// * recover is only partially implemented.  Also, the interpreter
// makes no attempt to distinguish target panics from interpreter
// crashes.
//
// * map iteration is asymptotically inefficient.
//
// * the equivalence relation for structs doesn't skip over blank
// fields.
//
// * the sizes of the int, uint and uintptr types in the target
// program are assumed to be the same as those of the interpreter
// itself.
//
// * os.Exit is implemented using panic, causing deferred functions to
// run.
package interp

import (
	"exp/ssa"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"os"
	"reflect"
	"runtime"
)

type status int

const (
	stRunning status = iota
	stComplete
	stPanic
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

// State shared between all interpreted goroutines.
type interpreter struct {
	prog           *ssa.Program         // the SSA program
	globals        map[ssa.Value]*value // addresses of global variables (immutable)
	mode           Mode                 // interpreter options
	reflectPackage *ssa.Package         // the fake reflect package
	rtypeMethods   ssa.MethodSet        // the method set of rtype, which implements the reflect.Type interface.
}

type frame struct {
	i                *interpreter
	caller           *frame
	fn               *ssa.Function
	block, prevBlock *ssa.BasicBlock
	env              map[ssa.Value]value // dynamic values of SSA variables
	locals           []value
	defers           []func()
	result           value
	status           status
	panic            interface{}
}

func (fr *frame) get(key ssa.Value) value {
	switch key := key.(type) {
	case nil:
		return nil
	case *ssa.Function, *ssa.Builtin:
		return key
	case *ssa.Literal:
		return literalValue(key)
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

// findMethodSet returns the method set for type typ, which may be one
// of the interpreter's fake types.
func findMethodSet(i *interpreter, typ types.Type) ssa.MethodSet {
	if typ == rtypeType {
		return i.rtypeMethods
	}
	return i.prog.MethodSet(typ)
}

// visitInstr interprets a single ssa.Instruction within the activation
// record frame.  It returns a continuation value indicating where to
// read the next instruction from.
func visitInstr(fr *frame, instr ssa.Instruction) continuation {
	switch instr := instr.(type) {
	case *ssa.UnOp:
		fr.env[instr] = unop(instr, fr.get(instr.X))

	case *ssa.BinOp:
		fr.env[instr] = binop(instr.Op, fr.get(instr.X), fr.get(instr.Y))

	case *ssa.Call:
		fn, args := prepareCall(fr, &instr.CallCommon)
		fr.env[instr] = call(fr.i, fr, instr.Pos, fn, args)

	case *ssa.Conv:
		fr.env[instr] = conv(instr.Type(), instr.X.Type(), fr.get(instr.X))

	case *ssa.ChangeInterface:
		x := fr.get(instr.X)
		if err := checkInterface(fr.i, instr.Type(), x.(iface)); err != "" {
			panic(err)
		}
		fr.env[instr] = x

	case *ssa.MakeInterface:
		fr.env[instr] = iface{t: instr.X.Type(), v: fr.get(instr.X)}

	case *ssa.Extract:
		fr.env[instr] = fr.get(instr.Tuple).(tuple)[instr.Index]

	case *ssa.Slice:
		fr.env[instr] = slice(fr.get(instr.X), fr.get(instr.Low), fr.get(instr.High))

	case *ssa.Ret:
		switch len(instr.Results) {
		case 0:
		case 1:
			fr.result = fr.get(instr.Results[0])
		default:
			var res []value
			for _, r := range instr.Results {
				res = append(res, copyVal(fr.get(r)))
			}
			fr.result = tuple(res)
		}
		return kReturn

	case *ssa.Panic:
		panic(targetPanic{fr.get(instr.X)})

	case *ssa.Send:
		fr.get(instr.Chan).(chan value) <- copyVal(fr.get(instr.X))

	case *ssa.Store:
		*fr.get(instr.Addr).(*value) = copyVal(fr.get(instr.Val))

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
		fn, args := prepareCall(fr, &instr.CallCommon)
		fr.defers = append(fr.defers, func() { call(fr.i, fr, instr.Pos, fn, args) })

	case *ssa.Go:
		fn, args := prepareCall(fr, &instr.CallCommon)
		go call(fr.i, nil, instr.Pos, fn, args)

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
		*addr = zero(indirectType(instr.Type()))

	case *ssa.MakeSlice:
		slice := make([]value, asInt(fr.get(instr.Cap)))
		tElt := underlyingType(instr.Type()).(*types.Slice).Elt
		for i := range slice {
			slice[i] = zero(tElt)
		}
		fr.env[instr] = slice[:asInt(fr.get(instr.Len))]

	case *ssa.MakeMap:
		reserve := 0
		if instr.Reserve != nil {
			reserve = asInt(fr.get(instr.Reserve))
		}
		fr.env[instr] = makeMap(underlyingType(instr.Type()).(*types.Map).Key, reserve)

	case *ssa.Range:
		fr.env[instr] = rangeIter(fr.get(instr.X), instr.X.Type())

	case *ssa.Next:
		fr.env[instr] = fr.get(instr.Iter).(iter).next()

	case *ssa.FieldAddr:
		x := fr.get(instr.X)
		fr.env[instr] = &(*x.(*value)).(structure)[instr.Field]

	case *ssa.Field:
		fr.env[instr] = copyVal(fr.get(instr.X).(structure)[instr.Field])

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
		fr.env[instr] = copyVal(fr.get(instr.X).(array)[asInt(fr.get(instr.Index))])

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
		for i, pred := range instr.Block_.Preds {
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
			if state.Dir == ast.RECV {
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
		var recvV value
		if recvOk {
			// No need to copy since send makes an unaliased copy.
			recvV = recv.Interface().(value)
		} else if chosen != -1 {
			// Ensure we provide a type-appropriate zero value.
			recvV = zero(underlyingType(instr.States[chosen].Chan.Type()).(*types.Chan).Elt)
		}
		fr.env[instr] = tuple{chosen, recvV, recvOk}

	default:
		panic(fmt.Sprintf("unexpected instruction: %T", instr))
	}

	// if val, ok := instr.(ssa.Value); ok {
	// 	fmt.Println(toString(fr.env[val])) // debugging
	// }

	return kNext
}

// prepareCall determines the function value and argument values for a
// function call in a Call, Go or Defer instruction, peforming
// interface method lookup if needed.
//
func prepareCall(fr *frame, call *ssa.CallCommon) (fn value, args []value) {
	if call.Func != nil {
		// Function call.
		fn = fr.get(call.Func)
	} else {
		// Interface method invocation.
		recv := fr.get(call.Recv).(iface)
		if recv.t == nil {
			panic("method invoked on nil interface")
		}
		meth := underlyingType(call.Recv.Type()).(*types.Interface).Methods[call.Method]
		id := ssa.IdFromQualifiedName(meth.QualifiedName)
		m := findMethodSet(fr.i, recv.t)[id]
		if m == nil {
			// Unreachable in well-typed programs.
			panic(fmt.Sprintf("method set for dynamic type %v does not contain %s", recv.t, id))
		}
		_, aptr := recv.v.(*value)                        // actual pointerness
		_, fptr := m.Signature.Recv.Type.(*types.Pointer) // formal pointerness
		switch {
		case aptr == fptr:
			args = append(args, copyVal(recv.v))
		case aptr:
			// Calling func(T) with a *T receiver: make a copy.
			args = append(args, copyVal(*recv.v.(*value)))
		case fptr:
			panic("illegal call of *T method with T receiver")
		}
		fn = m
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
		fset := fn.Prog.Files
		// TODO(adonovan): fix: loc() lies for external functions.
		fmt.Fprintf(os.Stderr, "Entering %s%s.\n", fn.FullName(), loc(fset, fn.Pos))
		suffix := ""
		if caller != nil {
			suffix = ", resuming " + caller.fn.FullName() + loc(fset, callpos)
		}
		defer fmt.Fprintf(os.Stderr, "Leaving %s%s.\n", fn.FullName(), suffix)
	}
	if fn.Enclosing == nil {
		name := fn.FullName()
		if ext := externals[name]; ext != nil {
			if i.mode&EnableTracing != 0 {
				fmt.Fprintln(os.Stderr, "\t(external)")
			}
			return ext(fn, args)
		}
		if fn.Blocks == nil {
			panic("no code for function: " + name)
		}
	}
	fr := &frame{
		i:      i,
		caller: caller, // currently unused; for unwinding.
		fn:     fn,
		env:    make(map[ssa.Value]value),
		block:  fn.Blocks[0],
		locals: make([]value, len(fn.Locals)),
	}
	for i, l := range fn.Locals {
		fr.locals[i] = zero(indirectType(l.Type()))
		fr.env[l] = &fr.locals[i]
	}
	for i, p := range fn.Params {
		fr.env[p] = args[i]
	}
	for i, fv := range fn.FreeVars {
		fr.env[fv] = env[i]
	}
	var instr ssa.Instruction

	defer func() {
		if fr.status != stComplete {
			if fr.i.mode&DisableRecover != 0 {
				return // let interpreter crash
			}
			fr.status, fr.panic = stPanic, recover()
		}
		for i := range fr.defers {
			if fr.i.mode&EnableTracing != 0 {
				fmt.Fprintln(os.Stderr, "Invoking deferred function", i)
			}
			fr.defers[len(fr.defers)-1-i]()
		}
		// Destroy the locals to avoid accidental use after return.
		for i := range fn.Locals {
			fr.locals[i] = bad{}
		}
		if fr.status == stPanic {
			panic(fr.panic) // panic stack is not entirely clean
		}
	}()

	for {
		if i.mode&EnableTracing != 0 {
			fmt.Fprintf(os.Stderr, ".%s:\n", fr.block)
		}
	block:
		for _, instr = range fr.block.Instrs {
			if i.mode&EnableTracing != 0 {
				if v, ok := instr.(ssa.Value); ok {
					fmt.Fprintln(os.Stderr, "\t", v.Name(), "=", instr)
				} else {
					fmt.Fprintln(os.Stderr, "\t", instr)
				}
			}
			switch visitInstr(fr, instr) {
			case kReturn:
				fr.status = stComplete
				return fr.result
			case kNext:
				// no-op
			case kJump:
				break block
			}
		}
	}
	panic("unreachable")
}

// setGlobal sets the value of a system-initialized global variable.
func setGlobal(i *interpreter, pkg *ssa.Package, name string, v value) {
	if g, ok := i.globals[pkg.Var(name)]; ok {
		*g = v
		return
	}
	panic("no global variable: " + pkg.Name() + "." + name)
}

// Interpret interprets the Go program whose main package is mainpkg.
// mode specifies various interpreter options.  filename and args are
// the initial values of os.Args for the target program.
//
// Interpret returns the exit code of the program: 2 for panic (like
// gc does), or the argument to os.Exit for normal termination.
//
func Interpret(mainpkg *ssa.Package, mode Mode, filename string, args []string) (exitCode int) {
	i := &interpreter{
		prog:    mainpkg.Prog,
		globals: make(map[ssa.Value]*value),
		mode:    mode,
	}
	initReflect(i)

	for importPath, pkg := range i.prog.Packages {
		// Initialize global storage.
		for _, m := range pkg.Members {
			switch v := m.(type) {
			case *ssa.Global:
				cell := zero(indirectType(v.Type()))
				i.globals[v] = &cell
			}
		}

		// Ad-hoc initialization for magic system variables.
		switch importPath {
		case "syscall":
			var envs []value
			for _, s := range os.Environ() {
				envs = append(envs, s)
			}
			envs = append(envs, "GOSSAINTERP=1")
			setGlobal(i, pkg, "envs", envs)

		case "runtime":
			// TODO(gri): expose go/types.sizeof so we can
			// avoid this fragile magic number;
			// unsafe.Sizeof(memStats) won't work since gc
			// and go/types have different sizeof
			// functions.
			setGlobal(i, pkg, "sizeof_C_MStats", uintptr(3696))

		case "os":
			Args := []value{filename}
			for _, s := range args {
				Args = append(Args, s)
			}
			setGlobal(i, pkg, "Args", Args)
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
			fmt.Fprintln(os.Stderr, "panic: unexpected type: %T", p)
		}

		// TODO(adonovan): dump panicking interpreter goroutine?
		// buf := make([]byte, 0x10000)
		// runtime.Stack(buf, false)
		// fmt.Fprintln(os.Stderr, string(buf))
		// (Or dump panicking target goroutine?)
	}()

	// Run!
	call(i, nil, token.NoPos, mainpkg.Init, nil)
	if mainFn := mainpkg.Func("main"); mainFn != nil {
		call(i, nil, token.NoPos, mainFn, nil)
		exitCode = 0
	} else {
		fmt.Fprintln(os.Stderr, "No main function.")
		exitCode = 1
	}
	return
}
