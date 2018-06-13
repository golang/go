// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

(() => {
	// Map web browser API and Node.js API to a single common API (preferring web standards over Node.js API).
	const isNodeJS = typeof process !== "undefined";
	if (isNodeJS) {
		global.require = require;
		global.fs = require("fs");

		const nodeCrypto = require("crypto");
		global.crypto = {
			getRandomValues(b) {
				nodeCrypto.randomFillSync(b);
			},
		};

		global.performance = {
			now() {
				const [sec, nsec] = process.hrtime();
				return sec * 1000 + nsec / 1000000;
			},
		};

		const util = require("util");
		global.TextEncoder = util.TextEncoder;
		global.TextDecoder = util.TextDecoder;
	} else {
		window.global = window;

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

	global.Go = class {
		constructor() {
			this.argv = ["js"];
			this.env = {};
			this.exit = (code) => {
				if (code !== 0) {
					console.warn("exit code:", code);
				}
			};

			const mem = () => {
				// The buffer may change when requesting more memory.
				return new DataView(this._inst.exports.mem.buffer);
			}

			const setInt64 = (addr, v) => {
				mem().setUint32(addr + 0, v, true);
				mem().setUint32(addr + 4, Math.floor(v / 4294967296), true);
			}

			const getInt64 = (addr) => {
				const low = mem().getUint32(addr + 0, true);
				const high = mem().getInt32(addr + 4, true);
				return low + high * 4294967296;
			}

			const loadValue = (addr) => {
				const id = mem().getUint32(addr, true);
				return this._values[id];
			}

			const storeValue = (addr, v) => {
				if (v === undefined) {
					mem().setUint32(addr, 0, true);
					return;
				}
				if (v === null) {
					mem().setUint32(addr, 1, true);
					return;
				}
				this._values.push(v);
				mem().setUint32(addr, this._values.length - 1, true);
			}

			const loadSlice = (addr) => {
				const array = getInt64(addr + 0);
				const len = getInt64(addr + 8);
				return new Uint8Array(this._inst.exports.mem.buffer, array, len);
			}

			const loadSliceOfValues = (addr) => {
				const array = getInt64(addr + 0);
				const len = getInt64(addr + 8);
				const a = new Array(len);
				for (let i = 0; i < len; i++) {
					const id = mem().getUint32(array + i * 4, true);
					a[i] = this._values[id];
				}
				return a;
			}

			const loadString = (addr) => {
				const saddr = getInt64(addr + 0);
				const len = getInt64(addr + 8);
				return decoder.decode(new DataView(this._inst.exports.mem.buffer, saddr, len));
			}

			const timeOrigin = Date.now() - performance.now();
			this.importObject = {
				go: {
					// func wasmExit(code int32)
					"runtime.wasmExit": (sp) => {
						this.exit(mem().getInt32(sp + 8, true));
					},

					// func wasmWrite(fd uintptr, p unsafe.Pointer, n int32)
					"runtime.wasmWrite": (sp) => {
						const fd = getInt64(sp + 8);
						const p = getInt64(sp + 16);
						const n = mem().getInt32(sp + 24, true);
						fs.writeSync(fd, new Uint8Array(this._inst.exports.mem.buffer, p, n));
					},

					// func nanotime() int64
					"runtime.nanotime": (sp) => {
						setInt64(sp + 8, (timeOrigin + performance.now()) * 1000000);
					},

					// func walltime() (sec int64, nsec int32)
					"runtime.walltime": (sp) => {
						const msec = (new Date).getTime();
						setInt64(sp + 8, msec / 1000);
						mem().setInt32(sp + 16, (msec % 1000) * 1000000, true);
					},

					// func getRandomData(r []byte)
					"runtime.getRandomData": (sp) => {
						crypto.getRandomValues(loadSlice(sp + 8));
					},

					// func boolVal(value bool) Value
					"syscall/js.boolVal": (sp) => {
						storeValue(sp + 16, mem().getUint8(sp + 8) !== 0);
					},

					// func intVal(value int) Value
					"syscall/js.intVal": (sp) => {
						storeValue(sp + 16, getInt64(sp + 8));
					},

					// func floatVal(value float64) Value
					"syscall/js.floatVal": (sp) => {
						storeValue(sp + 16, mem().getFloat64(sp + 8, true));
					},

					// func stringVal(value string) Value
					"syscall/js.stringVal": (sp) => {
						storeValue(sp + 24, loadString(sp + 8));
					},

					// func (v Value) Get(key string) Value
					"syscall/js.Value.Get": (sp) => {
						storeValue(sp + 32, Reflect.get(loadValue(sp + 8), loadString(sp + 16)));
					},

					// func (v Value) set(key string, value Value)
					"syscall/js.Value.set": (sp) => {
						Reflect.set(loadValue(sp + 8), loadString(sp + 16), loadValue(sp + 32));
					},

					// func (v Value) Index(i int) Value
					"syscall/js.Value.Index": (sp) => {
						storeValue(sp + 24, Reflect.get(loadValue(sp + 8), getInt64(sp + 16)));
					},

					// func (v Value) setIndex(i int, value Value)
					"syscall/js.Value.setIndex": (sp) => {
						Reflect.set(loadValue(sp + 8), getInt64(sp + 16), loadValue(sp + 24));
					},

					// func (v Value) call(name string, args []Value) (Value, bool)
					"syscall/js.Value.call": (sp) => {
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
					"syscall/js.Value.invoke": (sp) => {
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

					// func (v Value) new(args []Value) (Value, bool)
					"syscall/js.Value.new": (sp) => {
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
					"syscall/js.Value.Float": (sp) => {
						mem().setFloat64(sp + 16, parseFloat(loadValue(sp + 8)), true);
					},

					// func (v Value) Int() int
					"syscall/js.Value.Int": (sp) => {
						setInt64(sp + 16, parseInt(loadValue(sp + 8)));
					},

					// func (v Value) Bool() bool
					"syscall/js.Value.Bool": (sp) => {
						mem().setUint8(sp + 16, !!loadValue(sp + 8));
					},

					// func (v Value) Length() int
					"syscall/js.Value.Length": (sp) => {
						setInt64(sp + 16, parseInt(loadValue(sp + 8).length));
					},

					// func (v Value) prepareString() (Value, int)
					"syscall/js.Value.prepareString": (sp) => {
						const str = encoder.encode(String(loadValue(sp + 8)));
						storeValue(sp + 16, str);
						setInt64(sp + 24, str.length);
					},

					// func (v Value) loadString(b []byte)
					"syscall/js.Value.loadString": (sp) => {
						const str = loadValue(sp + 8);
						loadSlice(sp + 16).set(str);
					},

					"debug": (value) => {
						console.log(value);
					},
				}
			};
		}

		async run(instance) {
			this._inst = instance;
			this._values = [undefined, null, global, this._inst.exports.mem]; // TODO: garbage collection

			const mem = new DataView(this._inst.exports.mem.buffer)

			// Pass command line arguments and environment variables to WebAssembly by writing them to the linear memory.
			let offset = 4096;

			const strPtr = (str) => {
				let ptr = offset;
				new Uint8Array(mem.buffer, offset, str.length + 1).set(encoder.encode(str + "\0"));
				offset += str.length + (8 - (str.length % 8));
				return ptr;
			};

			const argc = this.argv.length;

			const argvPtrs = [];
			this.argv.forEach((arg) => {
				argvPtrs.push(strPtr(arg));
			});

			const keys = Object.keys(this.env).sort();
			argvPtrs.push(keys.length);
			keys.forEach((key) => {
				argvPtrs.push(strPtr(`${key}=${this.env[key]}`));
			});

			const argv = offset;
			argvPtrs.forEach((ptr) => {
				mem.setUint32(offset, ptr, true);
				mem.setUint32(offset + 4, 0, true);
				offset += 8;
			});

			this._inst.exports.run(argc, argv);
		}
	}

	if (isNodeJS) {
		if (process.argv.length < 3) {
			process.stderr.write("usage: go_js_wasm_exec [wasm binary] [arguments]\n");
			process.exit(1);
		}

		const go = new Go();
		go.argv = process.argv.slice(2);
		go.env = process.env;
		go.exit = process.exit;
		WebAssembly.instantiate(fs.readFileSync(process.argv[2]), go.importObject).then((result) => {
			return go.run(result.instance);
		}).catch((err) => {
			console.error(err);
			process.exit(1);
		});
	}
})();
