// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

let args = ["js"];

// Map web browser API and Node.js API to a single common API (preferring web standards over Node.js API).
if (typeof process !== "undefined") { // detect Node.js
	if (process.argv.length < 3) {
		process.stderr.write("usage: go_js_wasm_exec [wasm binary]\n");
		process.exit(1);
	}

	args = args.concat(process.argv.slice(3));
	global.require = require;

	global.fs = require("fs");

	const nodeCrypto = require("crypto");
	global.crypto = {
		getRandomValues(b) {
			nodeCrypto.randomFillSync(b);
		},
	};

	const now = () => {
		const [sec, nsec] = process.hrtime();
		return sec * 1000 + nsec / 1000000;
	};
	global.performance = {
		timeOrigin: Date.now() - now(),
		now: now,
	};

	const util = require("util");
	global.TextEncoder = util.TextEncoder;
	global.TextDecoder = util.TextDecoder;

	compileAndRun(fs.readFileSync(process.argv[2])).catch((err) => {
		console.error(err);
	});
} else {
	window.global = window;

	global.process = {
		env: {},
		exit(code) {
			if (code !== 0) {
				console.warn("exit code:", code);
			}
		},
	};

	let outputBuf = "";
	global.fs = {
		constants: {},
		writeSync(fd, buf) {
			outputBuf += decoder.decode(buf);
			const nl = outputBuf.lastIndexOf("\n");
			if (nl != -1) {
				console.log(outputBuf.substr(0, nl));
				outputBuf = outputBuf.substr(nl + 1);
			}
			return buf.length;
		},
	};
}

const encoder = new TextEncoder("utf-8");
const decoder = new TextDecoder("utf-8");

let mod, inst;
let values = []; // TODO: garbage collection

function mem() {
	// The buffer may change when requesting more memory.
	return new DataView(inst.exports.mem.buffer);
}

function setInt64(addr, v) {
	mem().setUint32(addr + 0, v, true);
	if (v >= 0) {
		mem().setUint32(addr + 4, v / 4294967296, true);
	} else {
		mem().setUint32(addr + 4, -1, true); // FIXME
	}
}

function getInt64(addr) {
	const low = mem().getUint32(addr + 0, true);
	const high = mem().getInt32(addr + 4, true);
	return low + high * 4294967296;
}

function loadValue(addr) {
	const id = mem().getUint32(addr, true);
	return values[id];
}

function storeValue(addr, v) {
	if (v === undefined) {
		mem().setUint32(addr, 0, true);
		return;
	}
	if (v === null) {
		mem().setUint32(addr, 1, true);
		return;
	}
	values.push(v);
	mem().setUint32(addr, values.length - 1, true);
}

function loadSlice(addr) {
	const array = getInt64(addr + 0);
	const len = getInt64(addr + 8);
	return new Uint8Array(inst.exports.mem.buffer, array, len);
}

function loadSliceOfValues(addr) {
	const array = getInt64(addr + 0);
	const len = getInt64(addr + 8);
	const a = new Array(len);
	for (let i = 0; i < len; i++) {
		const id = mem().getUint32(array + i * 4, true);
		a[i] = values[id];
	}
	return a;
}

function loadString(addr) {
	const saddr = getInt64(addr + 0);
	const len = getInt64(addr + 8);
	return decoder.decode(new DataView(inst.exports.mem.buffer, saddr, len));
}

async function compileAndRun(source) {
	await compile(source);
	await run();
}

async function compile(source) {
	mod = await WebAssembly.compile(source);
}

async function run() {
	let importObject = {
		js: {
			// func wasmexit(code int32)
			"runtime.wasmexit": function (sp) {
				process.exit(mem().getInt32(sp + 8, true));
			},

			// func wasmwrite(fd uintptr, p unsafe.Pointer, n int32)
			"runtime.wasmwrite": function (sp) {
				const fd = getInt64(sp + 8);
				const p = getInt64(sp + 16);
				const n = mem().getInt32(sp + 24, true);
				fs.writeSync(fd, new Uint8Array(inst.exports.mem.buffer, p, n));
			},

			// func nanotime() int64
			"runtime.nanotime": function (sp) {
				setInt64(sp + 8, (performance.timeOrigin + performance.now()) * 1000000);
			},

			// func walltime() (sec int64, nsec int32)
			"runtime.walltime": function (sp) {
				const msec = (new Date).getTime();
				setInt64(sp + 8, msec / 1000);
				mem().setInt32(sp + 16, (msec % 1000) * 1000000, true);
			},

			// func boolVal(value bool) Value
			"runtime/js.boolVal": function (sp) {
				storeValue(sp + 16, mem().getUint8(sp + 8) !== 0);
			},

			// func intVal(value int) Value
			"runtime/js.intVal": function (sp) {
				storeValue(sp + 16, getInt64(sp + 8));
			},

			// func floatVal(value float64) Value
			"runtime/js.floatVal": function (sp) {
				storeValue(sp + 16, mem().getFloat64(sp + 8, true));
			},

			// func stringVal(value string) Value
			"runtime/js.stringVal": function (sp) {
				storeValue(sp + 24, loadString(sp + 8));
			},

			// func (v Value) Get(key string) Value
			"runtime/js.Value.Get": function (sp) {
				storeValue(sp + 32, Reflect.get(loadValue(sp + 8), loadString(sp + 16)));
			},

			// func (v Value) set(key string, value Value)
			"runtime/js.Value.set": function (sp) {
				Reflect.set(loadValue(sp + 8), loadString(sp + 16), loadValue(sp + 32));
			},

			// func (v Value) Index(i int) Value
			"runtime/js.Value.Index": function (sp) {
				storeValue(sp + 24, Reflect.get(loadValue(sp + 8), getInt64(sp + 16)));
			},

			// func (v Value) setIndex(i int, value Value)
			"runtime/js.Value.setIndex": function (sp) {
				Reflect.set(loadValue(sp + 8), getInt64(sp + 16), loadValue(sp + 24));
			},

			// func (v Value) call(name string, args []Value) (Value, bool)
			"runtime/js.Value.call": function (sp) {
				try {
					const v = loadValue(sp + 8);
					const m = Reflect.get(v, loadString(sp + 16));
					const args = loadSliceOfValues(sp + 32);
					storeValue(sp + 56, Reflect.apply(m, v, args));
					mem().setUint8(sp + 60, 1);
				} catch (err) {
					storeValue(sp + 56, err);
					mem().setUint8(sp + 60, 0);
				}
			},

			// func (v Value) invoke(args []Value) (Value, bool)
			"runtime/js.Value.invoke": function (sp) {
				try {
					const v = loadValue(sp + 8);
					const args = loadSliceOfValues(sp + 16);
					storeValue(sp + 40, Reflect.apply(v, undefined, args));
					mem().setUint8(sp + 44, 1);
				} catch (err) {
					storeValue(sp + 40, err);
					mem().setUint8(sp + 44, 0);
				}
			},

			// func (v Value) wasmnew(args []Value) (Value, bool)
			"runtime/js.Value.wasmnew": function (sp) {
				try {
					const v = loadValue(sp + 8);
					const args = loadSliceOfValues(sp + 16);
					storeValue(sp + 40, Reflect.construct(v, args));
					mem().setUint8(sp + 44, 1);
				} catch (err) {
					storeValue(sp + 40, err);
					mem().setUint8(sp + 44, 0);
				}
			},

			// func (v Value) Float() float64
			"runtime/js.Value.Float": function (sp) {
				mem().setFloat64(sp + 16, parseFloat(loadValue(sp + 8)), true);
			},

			// func (v Value) Int() int
			"runtime/js.Value.Int": function (sp) {
				setInt64(sp + 16, parseInt(loadValue(sp + 8)));
			},

			// func (v Value) Bool() bool
			"runtime/js.Value.Bool": function (sp) {
				mem().setUint8(sp + 16, !!loadValue(sp + 8));
			},

			// func (v Value) Length() int
			"runtime/js.Value.Length": function (sp) {
				setInt64(sp + 16, parseInt(loadValue(sp + 8).length));
			},

			// func (v Value) prepareString() (Value, int)
			"runtime/js.Value.prepareString": function (sp) {
				const str = encoder.encode(String(loadValue(sp + 8)));
				storeValue(sp + 16, str);
				setInt64(sp + 24, str.length);
			},

			// func (v Value) loadString(b []byte)
			"runtime/js.Value.loadString": function (sp) {
				const str = loadValue(sp + 8);
				loadSlice(sp + 16).set(str);
			},

			"debug": function (value) {
				console.log(value);
			},
		}
	};

	inst = await WebAssembly.instantiate(mod, importObject);
	values = [undefined, null, global, inst.exports.mem];

	// Pass command line arguments and environment variables to WebAssembly by writing them to the linear memory.
	let offset = 4096;

	const strPtr = (str) => {
		let ptr = offset;
		new Uint8Array(inst.exports.mem.buffer, offset, str.length + 1).set(encoder.encode(str + "\0"));
		offset += str.length + (8 - (str.length % 8));
		return ptr;
	};

	const argc = args.length;

	const argvPtrs = [];
	args.forEach((arg) => {
		argvPtrs.push(strPtr(arg));
	});

	const keys = Object.keys(process.env).sort();
	argvPtrs.push(keys.length);
	keys.forEach((key) => {
		argvPtrs.push(strPtr(`${key}=${process.env[key]}`));
	});

	const argv = offset;
	argvPtrs.forEach((ptr) => {
		mem().setUint32(offset, ptr, true);
		mem().setUint32(offset + 4, 0, true);
		offset += 8;
	});

	try {
		inst.exports.run(argc, argv);
	} catch (err) {
		console.error(err);
		process.exit(1);
	}
}
